begin

fr=addfile("el_rn4ea.nc","r")
fs=addfile("el_slhf4ea.nc","r")
fp=addfile("el_pet4ea.nc","r")
ft=addfile("el_t2m4ea.nc","r")

y0=fr->rn(:,:,85:293,220:400)
y1=fs->slhf(:,:,85:293,220:400)
y2=fp->pet(:,:,85:293,220:400)
x=ft->t2m(:,:,85:293,220:400)

dim=dimsizes(y0)
year=dim(0)
day=dim(1)
nlat=dim(2)
nlon=dim(3)

;calculating ΔH

z1=y1(:,:,:,:)
z2=y2(:,:,:,:)
do t=0,year-1,1
   do d=0,day-1,1
      do i=0,nlat-1,1
         do j=0,nlon-1,1
            z1(t,d,i,j)=y0(t,d,i,j)-y1(t,d,i,j)
            z2(t,d,i,j)=y0(t,d,i,j)-y2(t,d,i,j)
         end do
      end do
   end do
end do

printVarSummary(z1)
setfileoption("nc","Format","NetCDF4")
system("rm -f el_h4icp.nc")
fh=addfile("el_h4icp.nc","c")
fh->h=z1

printVarSummary(z2)
setfileoption("nc","Format","NetCDF4")
system("rm -f el_hp4icp.nc")
fhp=addfile("el_hp4icp.nc","c")
fhp->hp=z2

;calculating Π

z1=dim_standardize_n_Wrap(z1,0,(/0,1/))
z2=dim_standardize_n_Wrap(z2,0,(/0,1/))
x=dim_standardize_n_Wrap(x,0,(/0,1/))

p=z1(:,:,:,:)
do t=0,year-1,1
   do d=0,day-1,1
      do i=0,nlat-1,1
         do j=0,nlon-1,1
            p(t,d,i,j)=(z1(t,d,i,j)-z2(t,d,i,j))*x(t,d,i,j);
         end do
      end do
   end do
end do

; patho="/mnt/e/discipulus/doctorial/laci/el/"
printVarSummary(p)
setfileoption("nc","Format","NetCDF4")
system("rm -f el_paip4icp.nc")
f=addfile("el_paip4icp.nc","c")
f->paip=p

end