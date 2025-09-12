begin

path="/users/panwei/discipulus/postdoc/0#llp-icp/era/data/sml1/"

fns=systemfunc("ls "+path+"daily.sml14ea.????.grib")
infile=addfiles(fns,"r")
ListSetType(infile,"join")

y0=infile[:]->SWVL1_GDS0_DBLY(:,:,::-1,:)
y=y0(:,:,85:293,220:400)

y!0="year"
y!1="day"
y!2="lat"
y!3="lon"

printVarSummary(y)
system("rm -f el_sml14icp.nc")
setfileoption("nc","Format","LargeFile")
fo=addfile("el_sml14icp.nc","c")
fo->sml1=y

end