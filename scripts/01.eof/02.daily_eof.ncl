load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "/users/panwei/discipulus/postdoc/0#llp-icp/paper/newGMS/shapefile_utils.ncl"

begin

path="/users/panwei/discipulus/postdoc/0#llp-icp/era/"

; Read sml1 data
in=addfile(path+"el_sml14icp.nc","r")
x=in->sml1(:,59:150,:,:)

; Apply GMS shapefile mask before EOF calculation
print("Applying GMS shapefile mask to sml1 data...")
opt_mask = True
opt_mask@keep = True  ; Keep data within GMS region
opt_mask@return_mask = False

; Apply mask for each year's data
x_masked = x
dims_x = dimsizes(x)
total_years = dims_x(0)
total_days = dims_x(1)

print("SML1 total years: " + total_years + ", total days: " + total_days)

do t = 0, total_years-1
   print("SML1 processing year " + (t+1979) + " (" + (t+1) + "/" + total_years + ")")
   do d = 0, total_days-1
      ; Apply mask to each time slice
      x_slice = x(t,d,:,:)
      x_slice!0 = "lat"
      x_slice!1 = "lon"
      
      ; Apply mask
      x_masked(t,d,:,:) = shapefile_mask_data(x_slice, "/users/panwei/discipulus/postdoc/0#llp-icp/paper/newGMS/gms.shp", opt_mask)
   end do
end do

print("SML1 mask processing completed!")

; Copy metadata
copy_VarMeta(x, x_masked)

; Detrend and standardize
print("SML1 performing detrend and standardization...")
; x_masked=dtrend_n(x_masked,False,0)
x_masked=dtrend_n(x_masked,False,1)
x_masked=dim_standardize_n_Wrap(x_masked,0,(/0,1/))

dims=dimsizes(x_masked)
ny=dims(0)
nd=dims(1)
nlat=dims(2)
nlon=dims(3)

print("SML1 starting EOF calculation (first mode only)...")
eof=new((/ny,1,nlat,nlon/),float)  ; Keep first mode only
eof_mode=new((/ny,1,nlat,nlon/),float)
ts=new((/ny,1,nd/),float)  ; Keep first mode only

do t=0,ny-1,1
     print("SML1 calculating EOF for year " + (t+1979) + "...")
     neof=1  ; Calculate first mode only
     optEOF=True
     optEOF@jopt=1
     eof_temp=eofunc_n_Wrap(x_masked(t,:,:,:),neof,optEOF,0)
     eof(t,0,:,:)=eof_temp(0,:,:)  ; Save first mode only
     eof_mode_temp=smth9_Wrap(eof_temp,0.5,0.5,True)
     eof_mode(t,0,:,:)=eof_mode_temp(0,:,:)

     optETS=True
     optETS@jopt=1
     ts_temp=eofunc_ts_n_Wrap(x_masked(t,:,:,:),eof_mode_temp,optETS,0)
     ts(t,0,:)=ts_temp(0,:)  ; Save first mode only
     ts(t,0,:)=dim_standardize_Wrap(ts(t,0,:),1)
end do

ts!0="year"
ts!1="eofs"
ts!2="time"
ts&eofs=(/1/)  ; First mode only
ts&time=ispan(0,nd-1,1)

; Copy metadata for EOF patterns
copy_VarMeta(x_masked(0,0,:,:),eof(0,0,:,:))

eof=eof*(-1.)
ts=ts*(-1.)

print("SML1 saving results...")
printVarSummary(eof)
system("rm -f mam_sml12eof4icp_gms.nc")
f0=addfile("mam_sml12eof4icp_gms.nc","c")
f0->eof=eof

printVarSummary(ts)
system("rm -f mam_sml12ts4icp_gms.nc")
f1=addfile("mam_sml12ts4icp_gms.nc","c")
f1->ts=ts

opt=True
pcvar=eof@pcvar
prinfo=True

sig_ev=eofunc_north(eof@eval,ny,prinfo)
print(sig_ev)

print("SML1 GMS shapefile EOF first mode calculation completed!")

end