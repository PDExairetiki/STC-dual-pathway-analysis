begin

paths="/users/panwei/discipulus/postdoc/0#llp-icp/era/data/snsr/"
patht="/users/panwei/discipulus/postdoc/0#llp-icp/era/data/sntr/"

fnss=systemfunc("ls "+paths+"daily.snsr4ea.????.grib")
infiles=addfiles(fnss,"r")
ListSetType(infiles,"join")

fnst=systemfunc("ls "+patht+"daily.sntr4ea.????.grib")
infilet=addfiles(fnst,"r")
ListSetType(infilet,"join")

ys=infiles[:]->SSR_GDS0_SFC(:,:,::-1,:)
yt=infilet[:]->STR_GDS0_SFC(:,:,::-1,:)

yt!0="year"
yt!1="day"
yt!2="lat"
yt!3="lon"

ys!0="year"
ys!1="day"
ys!2="lat"
ys!3="lon"

dim=dimsizes(ys)

ys=ys/(30.*60.*60.)
yt=yt/(30.*60.*60.)

ys@units="W m**-2"
yt@units="W m**-2"

; printVarSummary(ys)
; setfileoption("nc","Format","NetCDF4")
; system("rm -f el_snsr4ea.nc")
; fo0=addfile("el_snsr4ea.nc","c")
; fo0->snsr=ys
; ; 删除:fileattdef(fo0,False)

; printVarSummary(yt)
; setfileoption("nc","Format","NetCDF4")
; system("rm -f el_sntr4ea.nc")
; fo1=addfile("el_sntr4ea.nc","c")
; fo1->sntr=yt
; ; 删除:fileattdef(fo1,False)

;rn calculating
rn = ys + yt

copy_VarCoords(ys, rn)
rn@long_name="Net Radiation"
rn@units="W m**-2"

setfileoption("nc","Format","NetCDF4")
system("rm -f el_rn4ea.nc")

; 创建新文件并添加坐标变量
fo=addfile("el_rn4ea.nc","c")

; 写入带有坐标信息的变量
fo->rn=rn

printVarSummary(rn)
printMinMax(rn, False)

delete(fo0)
delete(fo1)
delete(fo)

end