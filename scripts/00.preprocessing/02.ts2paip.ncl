load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
begin

f0=addfile("el_paip4icp.nc","r")
x0=f0->paip
x0=dim_standardize_n_Wrap(x0,0,(/0,1/))
tc0=dim_avg_n_Wrap(x0,(/2,3/))
ts0=dim_avg_n_Wrap(tc0,0)

day0=ispan(0,364,1)

wks=gsn_open_wks("png","ts2paip4icp")

stringw=(/"(e)","(b)"/);,"(c)","(d)"

;Time Series

rts=True
rts@gsnDraw=False
rts@gsnFrame=False
rts@gsnScale=True
rts@gsnYRefLine=0.
rts@gsnYRefLineColor="black"
rts@gsnYRefLineDashPattern=1
;rts@gsnXYBarChart=True
;rts@gsnAboveYRefLineColor="red"
;rts@gsnBelowYRefLineColor="blue"
rts@xyDashPatterns=(/0,2/)
rts@xyLineThicknessF=2.

rts@vpHeightF=0.6
rts@vpWidthF=0.6

rts@tiYAxisString=""

rts@trYMinF=-2
rts@trYMaxF=2

rts@tmXTOn=False
rts@tmYROn=False
rts@tmXBLabelFontHeightF=0.02
rts@tmYLLabelFontHeightF=0.02

rts@tmYLMode="Explicit"
rts@tmYLValues=ispan(-20,20,5)/10.
rts@tmYLLabels=ispan(-20,20,5)/10.

rts@gsnLeftStringFontHeightF=0.02

rts@gsnRightString="STC - ~F33~p~";

rts@vpHeightF=0.3
rts@vpWidthF=0.6

rts0=rts
rts0@trXMinF=min(day0)-1
rts0@trXMaxF=max(day0)+1
rts0@tmXBMode="Explicit"
rts0@tmXBValues=(/0,31,59,90,120,151,181,212,243,273,304,334/)
rts0@tmXBLabels=(/"J","F","M","A","M","J","J","A","S","O","N","D"/)

rts0@gsnLeftString="(e) Regional mean"

rts0@xyLineColors=(/"black","red"/)

plot=gsn_csm_xy(wks,day0,ts0,rts0)

dum=new((/41/),graphic)
do t=0,40,1
  dum(t)=gsn_add_polyline(wks,plot,day0,tc0(t,:),rts0)
end do

draw(plot)
frame(wks)

end
