begin

path="/users/panwei/discipulus/postdoc/0#llp-icp/era/data/sshf/"

;data for rest

fns=systemfunc("ls "+path+"daily.sshf4ea.????.grib")
infile=addfiles(fns,"r")
ListSetType(infile,"join")

y=infile[:]->SSHF_GDS0_SFC(:,:,::-1,:)

;data combining

y!0="year"
y!1="day"
y!2="lat"
y!3="lon"

;data preprocessing

y=(-1)*y/(24.*60.*60.)
y@units="W m**-2"

setfileoption("nc","Format","NetCDF4")
system("rm -f el_sshf4ea.nc")
fo=addfile("el_sshf4ea.nc","c")
fo->sshf=y

printVarSummary(y)
printMinMax(y,False)

delete(fo)
end