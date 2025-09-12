begin

path="/users/panwei/discipulus/postdoc/0#llp-icp/era/data/pet/"

fns=systemfunc("ls "+path+"daily.pet4ea.????.grib")
infile=addfiles(fns,"r")
ListSetType(infile,"join")

x=infile[:]->PEV_GDS0_SFC(:,:,::-1,:)

x!0="year"
x!1="day"
x!2="lat"
x!3="lon"

; 检查原始数据范围
print("原始PEV_GDS0_SFC数据范围：")
printMinMax(x, False)

; 单位转换：从日累积量（米）转换为日平均通量（瓦特每平方米）
; ERA5-LAND PEV_GDS0_SFC是日累积的潜在蒸发量（米）
; 转换公式：W m^-2 = (m) × (ρ × L) / (24×3600 s)
; 其中 ρ = 1000 kg m^-3, L = 2.5×10^6 J kg^-1
; 注意：ERA5采用向下为正的约定，需要乘以-1来转换符号
x = x * (-1.0) * 1000.0 * 2.5e6 / (24.0 * 3600.0)
x@units="W m**-2"

; 检查转换后的数据统计
print("转换后的PET数据统计：")
printMinMax(x, False)

; 计算百分位数
x_1d = ndtooned(x)
x_1d@_FillValue = x@_FillValue
x_1d = where(x_1d.eq.x@_FillValue, x@_FillValue, x_1d)
x_valid = x_1d(ind(.not.ismissing(x_1d)))

print("数据统计信息：")
print("有效数据点数：" + dimsizes(x_valid))

; 计算分位数
x_sorted = dim_pqsort(x_valid, 2)
n = dimsizes(x_valid)
p95_idx = round(0.95*n-1, 0)
p99_idx = round(0.99*n-1, 0)
p999_idx = round(0.999*n-1, 0)

print("95%分位数：" + sprintf("%.2f", x_sorted(p95_idx)))
print("99%分位数：" + sprintf("%.2f", x_sorted(p99_idx)))
print("99.9%分位数：" + sprintf("%.2f", x_sorted(p999_idx)))

setfileoption("nc","Format","NetCDF4")
system("rm -f el_pet4ea.nc")
fo=addfile("el_pet4ea.nc","c")
fo->pet=x

printVarSummary(x)

end