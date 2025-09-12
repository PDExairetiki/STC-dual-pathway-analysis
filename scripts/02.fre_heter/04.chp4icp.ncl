begin

path="../../data/climatology/eof/"

gt=addfile(path+"mam_sml12ts4icp_gms.nc","r")
tss0=gt->ts(:,0,:)
tss0=dtrend_n(tss0, False, 0)
tss=dim_avg_n_Wrap(tss0,0)
; Compare the difference between tss and the original regional average time series, 
; execute if the same, multiply by -1 if different
tss=tss*(-1.)
print(tss)

ht=addfile(path+"mam_t2m2ts4icp_gms.nc","r")
tst0=ht->ts(:,0,:)
tst0=dtrend_n(tst0, False, 0)
tst=dim_avg_n_Wrap(tst0,0)

; Cross spectra calculating
iopt=1
jave=3
pct=0.10

csdof=specxy_anal(tss,tst,iopt,jave,pct)

; Plot settings
wks=gsn_open_wks("png","spe_sm2t4icp")

stringw=(/"(a)","(b)","(c)","(d)"/)
strings=(/"MAM"/)
plots=new(2,graphic)

rts=True
rts@gsnDraw=False
rts@gsnFrame=False
rts@gsnScale=True

rts@xyLineThicknessF=(/3,3/)
rts@xyLineColors=(/"black","red"/)
rts@xyDashPatterns=(/0,2/)

rts@tiYAxisString =""

rts@tmXTOn=False
rts@tmYROn=False
rts@tmXBLabelFontHeightF=0.02
rts@tmYLLabelFontHeightF=0.02

rts@gsnLeftStringFontHeightF=0.02

rts@vpHeightF=0.3
rts@vpWidthF=0.6

; Plot coherence
rts0=rts
rts0@gsnYRefLine=0.32
rts0@gsnYRefLineColor="black"
rts0@gsnYRefLineDashPattern=2
rts0@gsnAboveYRefLineColor="red"
rts0@trYMinF=0.
rts0@trYMaxF=1.
rts0@gsnLeftString=stringw(0)+strings(0)
rts0@gsnRightString="Coherence(SM, T)"
plots(0)=gsn_csm_xy(wks,1./csdof@frq, csdof@coher,rts0)

; Plot phase
rts1=rts
rts1@gsnLeftString=stringw(1)+strings(0)
rts1@gsnRightString="Phase(SM, T)"
rts1@gsnYRefLine=0
rts1@gsnAboveYRefLineColor="yellow"
rts1@gsnBelowYRefLineColor="blue"
rts1@trYMinF=-200.
rts1@trYMaxF=200.
plots(1)=gsn_csm_xy(wks,1./csdof@frq, csdof@phase,rts1)

; Panel settings
pnlres=True
pnlres@amJust="TopLeft"

; Draw panel
gsn_panel(wks,plots,(/1,2/),pnlres)

end