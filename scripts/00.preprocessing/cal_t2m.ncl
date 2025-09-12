begin

path="/users/panwei/discipulus/postdoc/0#llp-icp/era/data/t2m/"

fns=systemfunc("ls "+path+"daily.t2m4ea.????.grib")
infile=addfiles(fns,"r")
ListSetType(infile,"join")

y=infile[:]->2T_GDS0_SFC(:,:,::-1,:)

y!0="year"
y!1="day"
y!2="lat"
y!3="lon"

y=y-273.15
y@units="C"


; 删除旧文件
system("rm -f el_t2m4icp.nc")

; 使用NetCDF-4格式
setfileoption("nc","Format","NetCDF4")
setfileoption("nc","DeflateLevel",1)  ; 启用压缩

fo=addfile("el_t2m4icp.nc","c")
fo->t2m=y(:,:,85:293,220:400)


printVarSummary(y)
printMinMax(y, False)
end