load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "../../shapefiles/shapefile_utils.ncl"

begin

path="../../data/2019/eof/"

; Read sml1 data for the low-latitude plateau from April 20 to June 23, 2019
print("=== 2019 Low-latitude Plateau SM EOF Analysis ===")
print("Time range: April 20 to June 23, 2019 (Days 109-173)")
print("Spatial range: Low-latitude Plateau (21-29°N, 97-108°E)")

in=addfile(path+"el_sml14icp.nc","r")
x=in->sml1(40,109:173,125:205,50:160)  ; 2019, low-latitude plateau

; Apply GMS shapefile mask before EOF calculation
print("Applying GMS shapefile mask to sml1 data...")
opt_mask = True
opt_mask@keep = True  ; Keep data within GMS region
opt_mask@return_mask = False

; Mask processing for 2019 data
x_masked = x
dims_x = dimsizes(x)
total_days = dims_x(0)

print("Total SM days: " + total_days)

do d = 0, total_days-1
   ; Apply mask to each time slice
   x_slice = x(d,:,:)
   x_slice!0 = "lat"
   x_slice!1 = "lon"
   
   ; Apply mask
   x_masked(d,:,:) = shapefile_mask_data(x_slice, "../../shapefiles/gms.shp", opt_mask)
end do

print("SML1 mask processing completed!")

; Copy metadata
copy_VarMeta(x, x_masked)

; Detrend and standardize
print("SML1 detrending and standardizing...")
; x_masked=dtrend_n(x_masked,False,0)
x_masked=dtrend_n(x_masked,False,0)
x_masked=dim_standardize_n_Wrap(x_masked,0,0)

dims=dimsizes(x_masked)
nd=dims(0)
nlat=dims(1)
nlon=dims(2)

print("SM starting EOF calculation (only first mode)...")
eof=new((/1,nlat,nlon/),float)  ; Keep only the first mode
eof_mode=new((/1,nlat,nlon/),float)
ts=new((/1,nd/),float)  ; Keep only the first mode

neof=1  ; Calculate only the first mode
optEOF=True
optEOF@jopt=1
eof_temp=eofunc_n_Wrap(x_masked,neof,optEOF,0)
eof(0,:,:)=eof_temp(0,:,:)  ; Save only the first mode
eof_mode_temp=smth9_Wrap(eof_temp,0.5,0.5,True)
eof_mode(0,:,:)=eof_mode_temp(0,:,:)

optETS=True
optETS@jopt=1
ts_temp=eofunc_ts_n_Wrap(x_masked,eof_mode_temp,optETS,0)
ts(0,:)=ts_temp(0,:)  ; Save only the first mode
ts(0,:)=dim_standardize_Wrap(ts(0,:),0)


ts!0="eofs"
ts!1="time"
ts&eofs=(/1/)  ; Only the first mode
ts&time=ispan(0,nd-1,1)

; Copy metadata for EOF mode
copy_VarMeta(x_masked(0,:,:),eof(0,:,:))

eof=eof*(-1.)
ts=ts*(-1.)

print("SM saving results...")
printVarSummary(eof)
system("rm -f 2019_sm_eof.nc")
f0=addfile("2019_sm_eof.nc","c")
f0->eof=eof
f0->pcvar=eof@pcvar

printVarSummary(ts)
system("rm -f 2019_sm_ts.nc")
f1=addfile("2019_sm_ts.nc","c")
f1->ts=ts

opt=True
pcvar=eof@pcvar
prinfo=True

sig_ev=eofunc_north(eof@eval,1,prinfo)
print(sig_ev)

print("2019 Low-latitude Plateau SM EOF first mode calculation completed!")
print("PC1 variance contribution rate: " + sprintf("%.2f", eof@pcvar(0)) + "%")

end