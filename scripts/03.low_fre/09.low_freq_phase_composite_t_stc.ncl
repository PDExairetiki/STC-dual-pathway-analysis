load "../../shapefiles/shapefile_utils.ncl"

begin

path="../../data/climatology/filtered/"
pathl="../../data/climatology/origin/"

ft=addfile(path+"mam_t2m4icp_gms_filtered.nc","r")
t=ft->ts_low

f=addfile(pathl+"fmamj_t2m4icp.nc","r")
x=f->t2m(28:119,:,:)
lsm=where(x.ne.x@_FillValue,1,x@_FillValue)

fs=addfile(path+"mam_paip4icp_gms_filtered.nc","r")
s=fs->ts_low

lat=t&lat
lon=t&lon
nlat=dimsizes(lat)
nlon=dimsizes(lon)
dn=dimsizes(t&day)

t=dtrend_n(t,False,0)
t=t(:,:,:)*lsm
s=dtrend_n(s,False,0)
s=s(:,:,:)*lsm

opt_mask = True
opt_mask@keep = True  ; Keep data within GMS region
opt_mask@return_mask = False

do d = 0, dn-1
      ; Apply mask to each time slice
      t_slice = t(d,:,:)
      t_slice!0 = "lat"
      t_slice!1 = "lon"
      
      ; Apply mask
      t(d,:,:) = shapefile_mask_data(t_slice, "../../shapefiles/gms.shp", opt_mask)
end do

t2=dim_avg_n_Wrap(t(6:15,:,:),0)
t3=dim_avg_n_Wrap(t(16:25,:,:),0)
t4=dim_avg_n_Wrap(t(26:35,:,:),0)
t5=dim_avg_n_Wrap(t(36:45,:,:),0)
; t6=dim_avg_n_Wrap(t(40:,:,:),0)

s=dim_standardize_n_Wrap(s,0,0)
s2=dim_avg_n_Wrap(s(6:15,:,:),0)
s3=dim_avg_n_Wrap(s(16:25,:,:),0)
s4=dim_avg_n_Wrap(s(26:35,:,:),0)
s5=dim_avg_n_Wrap(s(36:45,:,:),0)
; s6=dim_avg_n_Wrap(s(40:,:,:),0)
;time series

sm=new((/4,nlat,nlon/),float)
sm(0,:,:)=s2
sm(1,:,:)=s3
sm(2,:,:)=s4
sm(3,:,:)=s5
; sm(4,:,:)=s6

tm=new((/4,nlat,nlon/),float)
tm(0,:,:)=t2
tm(1,:,:)=t3
tm(2,:,:)=t4
tm(3,:,:)=t5
; tm(4,:,:)=t6

; 6. Perform significance test (t-test)
print("6. Performing significance test...")
alpha = 0.10  ; 90% confidence level

; Create significance arrays
significance_t = new((/4,nlat,nlon/), float)
significance_stc = new((/4,nlat,nlon/), float)
significance_t@_FillValue = t@_FillValue
significance_stc@_FillValue = s@_FillValue

; Perform t-test for each phase
do phase = 0, 3
    ; Calculate sample size for each phase (10 days)
    n_samples = 10
    
    ; Perform t-test for each grid point
    do i = 0, nlat-1
        do j = 0, nlon-1
            ; T t-test
            if (.not. ismissing(t(6+phase*10, i, j))) then
                ; Collect samples for this phase
                phase_samples_t = t(6+phase*10:15+phase*10, i, j)
                if (.not. all(ismissing(phase_samples_t))) then
                    t_stat = ttest(phase_samples_t, 0.0, n_samples, 0.0, n_samples, False, 0, 0)
                    significance_t(phase, i, j) = t_stat(0)  ; p-value
                end if
            end if
            
            ; STC t-test
            if (.not. ismissing(s(6+phase*10, i, j))) then
                phase_samples_stc = s(6+phase*10:15+phase*10, i, j)
                if (.not. all(ismissing(phase_samples_stc))) then
                    t_stat = ttest(phase_samples_stc, 0.0, n_samples, 0.0, n_samples, False, 0, 0)
                    significance_stc(phase, i, j) = t_stat(0)  ; p-value
                end if
            end if
        end do
    end do
end do

day=ispan(0,dn-1,1)

wks=gsn_open_wks("png","evo_esv_spa2stc4icp_gms0")

stringw=(/"(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)","(i)","(j)","(k)","(l)","(m)","(n)","(o)","(p)"/);
stringt=(/"Phase -3","Phase -2","Phase -1","Phase 0"/)
strings=(/"T~B~2mL~N~","SM~B~L~N~","STC - ~F33~p~FB~L~N~"/)
plot=new(8,graphic)
; plott=new(5,graphic)

res=True
res@gsnDraw=False
res@gsnFrame=False
res@gsnLeftString=""
res@gsnRightString=""
res@gsnSpreadColors=True
res@gsnAddCyclic=False

res@cnInfoLabelOn=False
res@cnLinesOn=False
res@cnLineLabelsOn=False
res@cnInfoLabelOn=False
res@cnFillOn=True
res@cnFillMode="AreaFill"
res@cnLinesOn=False
res@mpFillOn=True
res@lbLabelBarOn=False

res@mpMinLatF=lat(0)
res@mpMaxLatF=lat(nlat-1)
res@mpMinLonF=lon(1)
res@mpMaxLonF=lon(nlon-1)
; Set ocean fill
res@mpFillOn=True
res@mpOceanFillColor="gray90"
res@mpLandFillColor="gray90"
res@mpInlandWaterFillColor="gray90"
res@cnMissingValFillColor="gray90"
res@mpShapeMode="FreeAspect"
res@mpDataBaseVersion = "Ncarg4_1"
res@mpDataSetName = "Earth..4"
res@mpOutlineBoundarySets = "National"
res@mpOutlineSpecifiers = (/"China:states"/)

res@tmXTOn=False
res@tmYROn=False
; res@tmXBLabelFontHeightF=0.02
; res@tmYLLabelFontHeightF=0.02

res@vpHeightF=0.6
res@vpWidthF=0.6

res@pmTickMarkDisplayMode="Always"

res@cnLevelSelectionMode="ManualLevels"
res@cnMinLevelValF       = -1             ; min level
res@cnMaxLevelValF       = 1          ; max level
res@cnLevelSpacingF      = 0.1             ; contour interval
res@mpLandFillColor      = "white"
FillColors=(/20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240/)
;(/240,230,220,210,200,190,180,170,160,150,140,130,120,110,100,90,80,70,60,50,40,30,20/)
res@cnFillColors=FillColors
res@gsnLeftStringFontHeightF=0.02

do i=0,3,1
   res@gsnLeftString=stringw(i+4)+stringt(i)
   res@gsnRightString=strings(0)
   
   ; Show only areas that pass significance test
   tm_sig = where(significance_t(i, :, :) .lt. alpha, tm(i, :, :), tm@_FillValue)
   plot(i)=gsn_csm_contour_map_ce(wks,tm_sig,res)

   res@gsnLeftString=stringw(i+8)+stringt(i)
   res@gsnRightString=strings(2);+"-"+strings(3)
   
   ; Show only areas that pass significance test
   sm_sig = where(significance_stc(i, :, :) .lt. alpha, sm(i, :, :), sm@_FillValue)
   plot(i+4)=gsn_csm_contour_map_ce(wks,sm_sig,res)
end do

pnlres=True
pnlres@amJust="TopLeft"
;pnlres@gsnPanelYF=(/-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0.25,-1,-1,-1/)
;pnlres@gsnPanelXF=(/-1,-1,-1,-1,-1,-1,-1,-1,0.05,-1,-1,-1/)

pnlres@gsnPanelLabelBar=True
pnlres@lbLabelFontHeightF=0.007
;pnlres@pmLabelBarHeightF=0.08

gsn_panel(wks,plot,(/2,4/),pnlres)

end