load "../../shapefiles/shapefile_utils.ncl"

begin

path="../../data/climatology/filtered/"
pathl="../../data/climatology/origin/"

f=addfile(pathl+"fmamj_slhf4icp.nc","r")
x=f->slhf(28:119,:,:)
lsm=where(x.ne.x@_FillValue,1,x@_FillValue)

fs0=addfile(path+"mam_slhf4icp_gms_filtered.nc","r")
lh=fs0->ts_low

fs1=addfile(path+"mam_sshf4icp_gms_filtered.nc","r")
sh=fs1->ts_low

fs2=addfile(path+"mam_rn4icp_gms_filtered.nc","r")
rn=fs2->ts_low

lat=lh&lat
lon=lh&lon
nlat=dimsizes(lat)
nlon=dimsizes(lon)
dn=dimsizes(lh&day)

lh=dtrend_n(lh,False,0)
lh=lh(:,:,:)*lsm
sh=dtrend_n(sh,False,0)
sh=sh(:,:,:)*lsm
rn=dtrend_n(rn,False,0)
rn=rn(:,:,:)*lsm

opt_mask = True
opt_mask@keep = True  ; Keep data within GMS region
opt_mask@return_mask = False

do d = 0, dn-1
      ; Apply mask to each time slice
      lh_slice = lh(d,:,:)
      lh_slice!0 = "lat"
      lh_slice!1 = "lon"
      
      sh_slice = sh(d,:,:)
      sh_slice!0 = "lat"
      sh_slice!1 = "lon"

      rn_slice = rn(d,:,:)
      rn_slice!0 = "lat"
      rn_slice!1 = "lon"

      ; Apply mask
      lh_slice = shapefile_mask_data(lh_slice, "../../shapefiles/gms.shp", opt_mask)
      sh_slice = shapefile_mask_data(sh_slice, "../../shapefiles/gms.shp", opt_mask)
      rn_slice = shapefile_mask_data(rn_slice, "../../shapefiles/gms.shp", opt_mask)
end do

lh=dim_standardize_n_Wrap(lh,0,0)
lh0=dim_avg_n_Wrap(lh(6:15,:,:),0)
lh1=dim_avg_n_Wrap(lh(16:25,:,:),0)
lh2=dim_avg_n_Wrap(lh(26:35,:,:),0)
lh3=dim_avg_n_Wrap(lh(36:45,:,:),0)
; t6=dim_avg_n_Wrap(t(40:,:,:),0)

sh=dim_standardize_n_Wrap(sh,0,0)
sh0=dim_avg_n_Wrap(sh(6:15,:,:),0)
sh1=dim_avg_n_Wrap(sh(16:25,:,:),0)
sh2=dim_avg_n_Wrap(sh(26:35,:,:),0)
sh3=dim_avg_n_Wrap(sh(36:45,:,:),0)
; s6=dim_avg_n_Wrap(s(40:,:,:),0)

rn=dim_standardize_n_Wrap(rn,0,0)
rn0=dim_avg_n_Wrap(rn(6:15,:,:),0)
rn1=dim_avg_n_Wrap(rn(16:25,:,:),0)
rn2=dim_avg_n_Wrap(rn(26:35,:,:),0)
rn3=dim_avg_n_Wrap(rn(36:45,:,:),0)
;time series

sm=new((/4,nlat,nlon/),float)
sm(0,:,:)=lh0
sm(1,:,:)=lh1
sm(2,:,:)=lh2
sm(3,:,:)=lh3
; sm(4,:,:)=s6

tm=new((/4,nlat,nlon/),float)
tm(0,:,:)=sh0
tm(1,:,:)=sh1
tm(2,:,:)=sh2
tm(3,:,:)=sh3
; tm(4,:,:)=t6

pm=new((/4,nlat,nlon/),float)
pm(0,:,:)=rn0
pm(1,:,:)=rn1
pm(2,:,:)=rn2
pm(3,:,:)=rn3

; 6. Perform significance test (t-test)
print("6. Performing significance test...")
alpha = 0.10  ; 90% confidence level

; Create significance arrays
significance_lh = new((/4,nlat,nlon/), float)
significance_sh = new((/4,nlat,nlon/), float)
significance_rn = new((/4,nlat,nlon/), float)
significance_lh@_FillValue = lh@_FillValue
significance_sh@_FillValue = sh@_FillValue
significance_rn@_FillValue = rn@_FillValue

; Perform t-test for each phase
do phase = 0, 3
    ; Calculate sample size for each phase (10 days)
    n_samples = 10
    
    ; Perform t-test for each grid point
    do i = 0, nlat-1
        do j = 0, nlon-1
            ; Latent heat flux t-test
            if (.not. ismissing(lh(6+phase*10, i, j))) then
                ; Collect samples for this phase
                phase_samples_lh = lh(6+phase*10:15+phase*10, i, j)
                if (.not. all(ismissing(phase_samples_lh))) then
                    t_stat = ttest(phase_samples_lh, 0.0, n_samples, 0.0, n_samples, False, 0, 0)
                    significance_lh(phase, i, j) = t_stat(0)  ; p-value
                end if
            end if
            
            ; Sensible heat flux t-test
            if (.not. ismissing(sh(6+phase*10, i, j))) then
                phase_samples_sh = sh(6+phase*10:15+phase*10, i, j)
                if (.not. all(ismissing(phase_samples_sh))) then
                    t_stat = ttest(phase_samples_sh, 0.0, n_samples, 0.0, n_samples, False, 0, 0)
                    significance_sh(phase, i, j) = t_stat(0)  ; p-value
                end if
            end if
            
            ; Net radiation t-test
            if (.not. ismissing(rn(6+phase*10, i, j))) then
                phase_samples_rn = rn(6+phase*10:15+phase*10, i, j)
                if (.not. all(ismissing(phase_samples_rn))) then
                    t_stat = ttest(phase_samples_rn, 0.0, n_samples, 0.0, n_samples, False, 0, 0)
                    significance_rn(phase, i, j) = t_stat(0)  ; p-value
                end if
            end if
        end do
    end do
end do

day=ispan(0,dn-1,1)

wks=gsn_open_wks("png","evo_esv_spa2shf4icp_gms")

stringw=(/"(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)","(i)","(j)","(k)","(l)","(m)","(n)","(o)","(p)"/);
stringt=(/"Phase -3","Phase -2","Phase -1","Phase 0"/)
strings=(/"~F33~l~F~E~B~L~N","SH~B~L~N~","R~B~nL~N~"/)
plot=new(12,graphic)
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
   res@gsnLeftString=stringw(i)+stringt(i)
   res@gsnRightString=strings(0)
   
   ; Show only areas that pass significance test
   sm_sig = where(significance_lh(i, :, :) .lt. alpha, sm(i, :, :), sm@_FillValue)
   plot(i)=gsn_csm_contour_map_ce(wks,sm_sig,res)

   res@gsnLeftString=stringw(i+4)+stringt(i)
   res@gsnRightString=strings(1);+"-"+strings(3)
   
   ; Show only areas that pass significance test
   tm_sig = where(significance_sh(i, :, :) .lt. alpha, tm(i, :, :), tm@_FillValue)
   plot(i+4)=gsn_csm_contour_map_ce(wks,tm_sig,res)

   res@gsnLeftString=stringw(i+8)+stringt(i)
   res@gsnRightString=strings(2);+"-"+strings(3)
   
   ; Show only areas that pass significance test
   pm_sig = where(significance_rn(i, :, :) .lt. alpha, pm(i, :, :), pm@_FillValue)
   plot(i+8)=gsn_csm_contour_map_ce(wks,pm_sig,res)
end do

pnlres=True
pnlres@amJust="TopLeft"
;pnlres@gsnPanelYF=(/-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0.25,-1,-1,-1/)
;pnlres@gsnPanelXF=(/-1,-1,-1,-1,-1,-1,-1,-1,0.05,-1,-1,-1/)

pnlres@gsnPanelLabelBar=True
pnlres@lbLabelFontHeightF=0.007
;pnlres@pmLabelBarHeightF=0.08

gsn_panel(wks,plot,(/3,4/),pnlres)

end