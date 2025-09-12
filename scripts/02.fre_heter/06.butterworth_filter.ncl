begin

; Read projected time series
gt=addfile("mam_sml12ts4icp_gms_padded.nc","r")
tss_padded=gt->ts_padded
n_pad=gt->padding_length

ht=addfile("mam_t2m2ts4icp_gms_padded.nc","r")
tst_padded=ht->ts_padded

print("=== Butterworth Filter Design ===")
print("Projected time series length: " + dimsizes(tss_padded))

; Design filter parameters based on image analysis
; Image shows:
; - High frequency region: frequency 0-10, negative phase (SM lags T, temperature affects soil moisture)
; - Low frequency region: frequency 15-40, positive phase (SM leads T, soil moisture affects temperature)
; - Ignored: frequency 40+, phase near 0

; Filter parameter settings
dt = 1.0  ; Time interval (days)
nyquist = 1.0/(2.0*dt)  ; Nyquist frequency = 0.5

; High/low frequency boundary frequency: based on phase transition point analysis
; According to image analysis, the key transition point where phase changes from negative to positive is approximately at x=15-18
; Corresponding frequency is about 0.067 (period 15 days): phase changes from negative to positive, used as the high/low frequency boundary point
freq_cutoff = 0.067

; Low-pass filter: preserve frequency < 0.067 (period > 15 days, soil moisture affects temperature)
freq_low = freq_cutoff

; High-pass filter: preserve frequency > 0.067 (period < 15 days, temperature affects soil moisture)
freq_high = freq_cutoff

; Filter order
n_order = 4

print("")
print("Filter parameters:")
print("Nyquist frequency: " + sprintf("%.3f", nyquist))
print("Boundary frequency: " + sprintf("%.3f", freq_cutoff) + " (period " + sprintf("%.1f", 1.0/freq_cutoff) + " days)")
print("Low frequency: frequency < " + sprintf("%.3f", freq_cutoff) + " (soil moisture affects temperature)")
print("High frequency: frequency > " + sprintf("%.3f", freq_cutoff) + " (temperature affects soil moisture)")

; Before filtering, multiply SM sequence by -1 (opposite to original regional average, sign correction needed)
tss_padded = tss_padded * (-1.0)

; Apply Butterworth filter
; First print input data information
print("Input data statistics:")
print("  tss_padded range: " + min(tss_padded) + " to " + max(tss_padded))
print("  tst_padded range: " + min(tst_padded) + " to " + max(tst_padded))

; Use Butterworth bandpass function to implement low-pass/high-pass
opt = True
opt@m = 4                    ; Filter order
opt@dt = dt                  ; Sampling interval (days)
opt@remove_mean = False      ; Do not remove mean (avoid NaN)
opt@return_filtered = True
opt@return_envelope = False

; Low-pass: from minimum resolvable frequency to freq_cutoff
n_all = dimsizes(tss_padded)
eps = 1.0e-6
fmin = 1.0/(n_all*dt) + eps
fca_low = fmin
fcb_low = freq_cutoff
print("Applying low-frequency filter (Butterworth low-pass, near-zero phase bidirectional)...")
; Forward filtering
low_fwd_s = bw_bandpass_filter(tss_padded, fca_low, fcb_low, opt, 0)
low_fwd_t = bw_bandpass_filter(tst_padded, fca_low, fcb_low, opt, 0)
; Backward filtering
low_bwd_s = bw_bandpass_filter(tss_padded(::-1), fca_low, fcb_low, opt, 0)
low_bwd_t = bw_bandpass_filter(tst_padded(::-1), fca_low, fcb_low, opt, 0)
; Synthesize near-zero phase result (average of forward and backward results and reverse back to original order)
tss_low = 0.5*(low_fwd_s + low_bwd_s(::-1))
tst_low = 0.5*(low_fwd_t + low_bwd_t(::-1))
print("Low-frequency filtered data statistics:")
print("  tss_low range: " + min(tss_low) + " to " + max(tss_low))
print("  tst_low range: " + min(tst_low) + " to " + max(tst_low))

; High frequency uses complementary: original - low frequency, avoid boundary band fluctuations and ensure consistency with PC1
print("Calculating high-frequency component (original - low frequency)...")
tss_high = tss_padded - tss_low
tst_high = tst_padded - tst_low
print("High-frequency filtered data statistics:")
print("  tss_high range: " + min(tss_high) + " to " + max(tss_high))
print("  tst_high range: " + min(tst_high) + " to " + max(tst_high))

; Remove projected part, keep original length
; Projection structure: 30(left padding) + 92(original) + 30(right padding) = 152 days
; Need to extract middle 92 days of original data
original_length = 92
tss_low_orig = tss_low(n_pad:n_pad+original_length-1)    ; Index 30 to 121, 92 days total
tst_low_orig = tst_low(n_pad:n_pad+original_length-1)    ; Index 30 to 121, 92 days total

tss_high_orig = tss_high(n_pad:n_pad+original_length-1)  ; Index 30 to 121, 92 days total
tst_high_orig = tss_high(n_pad:n_pad+original_length-1)  ; Index 30 to 121, 92 days total

; Add time coordinates
time_orig = ispan(1, dimsizes(tss_low_orig), 1)
tss_low_orig!0 = "time"
tss_low_orig&time = time_orig
tst_low_orig!0 = "time"
tst_low_orig&time = time_orig

tss_high_orig!0 = "time"
tss_high_orig&time = time_orig
tst_high_orig!0 = "time"
tst_high_orig&time = time_orig

; Save filtered time series
system("rm -f mam_sml12ts4icp_gms_filtered.nc")
f1 = addfile("mam_sml12ts4icp_gms_filtered.nc", "c")
f1->ts_low = ts_low
f1->ts_high = ts_high

system("rm -f mam_t2m2ts4icp_gms_filtered.nc")
f2 = addfile("mam_t2m2ts4icp_gms_filtered.nc", "c")
f2->ts_low = tst_low
f2->ts_high = tst_high

print("")
print("Filtering completed!")
print("Saved files:")
print("  SML1: mam_sml12ts4icp_gms_filtered.nc")
print("  T2M:  mam_t2m2ts4icp_gms_filtered.nc")
print("")
print("Filtered results included:")
print("  ts_low: low-frequency component (period > " + sprintf("%.1f", 1.0/freq_cutoff) + " days, soil moisture affects temperature)")
print("  ts_high: high-frequency component (period < " + sprintf("%.1f", 1.0/freq_cutoff) + " days, temperature affects soil moisture)")

end