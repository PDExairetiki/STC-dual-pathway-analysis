begin

; Read first mode time series data
gt=addfile("mam_sml12ts4icp_gms.nc","r")
tss0=gt->ts(:,0,:)
tss0=dtrend_n(tss0, False, 0)
tss=dim_avg_n_Wrap(tss0,0)
tss=dim_standardize_n_Wrap(tss, 0, 0)

ht=addfile("mam_t2m2ts4icp_gms.nc","r")
tst0=ht->ts(:,0,:)
tst0=dtrend_n(tst0, False, 0)
tst=dim_avg_n_Wrap(tst0,0)
tst=dim_standardize_n_Wrap(tst, 0, 0)

print("=== Time Series Projection Processing ===")
print("Original time series length: " + dimsizes(tss))

; Projection method: Mirror projection (recommended for filtering)
; Target structure: 30(left padding) + 92(original) + 30(right padding) = 152 days
n_pad = 30  ; Projection length

; Apply mirror projection to SML1 time series
total_length = dimsizes(tss) + 2*n_pad  ; 92 + 60 = 152
tss_padded = new(total_length, float)
tss_padded!0 = "time"
tss_padded&time = ispan(0, total_length-1, 1)

; Fill original data in middle position
tss_padded(n_pad:n_pad+dimsizes(tss)-1) = tss

; Mirror projection: left end (index 0-29)
do i = 0, n_pad-1
   tss_padded(i) = tss(n_pad-1-i)
end do

; Mirror projection: right end (index 122-151)
do i = 0, n_pad-1
   tss_padded(n_pad+dimsizes(tss)+i) = tss(dimsizes(tss)-1-i)
end do

; Apply mirror projection to T2M time series
tst_padded = new(total_length, float)
tst_padded!0 = "time"
tst_padded&time = ispan(0, total_length-1, 1)

; Fill original data in middle position
tst_padded(n_pad:n_pad+dimsizes(tst)-1) = tst

; Mirror projection: left end (index 0-29)
do i = 0, n_pad-1
   tst_padded(i) = tst(n_pad-1-i)
end do

; Mirror projection: right end (index 122-151)
do i = 0, n_pad-1
   tst_padded(n_pad+dimsizes(tst)+i) = tst(dimsizes(tst)-1-i)
end do

print("Projected time series length: " + dimsizes(tss_padded))
print("Projection structure: " + n_pad + "(left padding) + " + dimsizes(tss) + "(original) + " + n_pad + "(right padding) = " + dimsizes(tss_padded))

; Save projected time series
system("rm -f mam_sml12ts4icp_gms_padded.nc")
f1 = addfile("mam_sml12ts4icp_gms_padded.nc", "c")
f1->ts_padded = tss_padded
f1->original_length = dimsizes(tss)
f1->padding_length = n_pad

system("rm -f mam_t2m2ts4icp_gms_padded.nc")
f2 = addfile("mam_t2m2ts4icp_gms_padded.nc", "c")
f2->ts_padded = tst_padded
f2->original_length = dimsizes(tst)
f2->padding_length = n_pad

print("")
print("Projected data saved to:")
print("  SML1: mam_sml12ts4icp_gms_padded.nc")
print("  T2M:  mam_t2m2ts4icp_gms_padded.nc")

end
