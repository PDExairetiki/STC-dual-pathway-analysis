begin

path="/users/panwei/discipulus/postdoc/0#llp-icp/era/data/slhf/"

;data for rest

fns=systemfunc("ls "+path+"daily.slhf4ea.????.grib")
infile=addfiles(fns,"r")
ListSetType(infile,"join")

y=infile[:]->SLHF_GDS0_SFC(:,:,::-1,:)

;data combining

y!0="year"
y!1="day"
y!2="lat"
y!3="lon"

;data preprocessing

y=(-1)*y/(24.*60.*60.)
y@units="W m**-2"

setfileoption("nc","Format","NetCDF4")
system("rm -f el_slhf4ea.nc")
fo=addfile("el_slhf4ea.nc","c")
fo->slhf=y

printVarSummary(y)
printMinMax(y,False)

delete(fo)
end