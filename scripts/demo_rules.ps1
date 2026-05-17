# demo_rules.ps1 -- One fraud scenario per rule, R01-R15
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\demo_rules.ps1
#
# All velocity/count checks are PRE-EVENT (computed before the current
# transaction is ingested).  A rule that fires when count >= N needs
# N prior transactions in the store -- the (N+1)th tx is the trigger.
#
# A run-specific suffix is appended to every account/device ID so each
# run starts with a clean feature-store slate on the remote server.

$URL = "https://aml-intelligence-platform.onrender.com/stream/transaction"
$HDR = @{ "Content-Type" = "application/json" }

$RUN = [System.Guid]::NewGuid().ToString("N").Substring(0, 4).ToUpper()
Write-Host ""
Write-Host "  Run ID: $RUN  (suffix on all IDs -- fresh state every run)" -ForegroundColor DarkGray

# ------------------------------------------------------------------
# Helper: build a complete 28-field TransactionEvent hashtable
# ------------------------------------------------------------------
function Tx {
    param(
        [string]$id,
        [string]$sender,
        [string]$receiver,
        [double]$amount,
        [string]$ts,
        [string]$type = "NEFT",
        [string]$channel = "Web",
        [string]$srcCountry = "IN",
        [string]$dstCountry = "IN",
        [bool]$intl = $false,
        [int]$kyc = 2,
        [double]$lat = 20.0,
        [double]$lon = 78.0,
        [string]$device = "",
        [string]$remarks = ""
    )
    $bal = [double](Get-Random -Min 500000 -Max 5000000)
    return @{
        transaction_id          = "${id}_${RUN}"
        sender_account          = "${sender}_${RUN}"
        receiver_account        = "${receiver}_${RUN}"
        sender_name             = "Demo_${sender}"
        receiver_name           = "Demo_${receiver}"
        amount                  = $amount
        currency                = "INR"
        timestamp               = $ts
        transaction_type        = $type
        payment_channel         = $channel
        sender_country          = $srcCountry
        receiver_country        = $dstCountry
        sender_bank             = "HDFC"
        receiver_bank           = "SBI"
        transaction_status      = "Success"
        is_international        = $intl
        kyc_level               = $kyc
        geo_latitude            = $lat
        geo_longitude           = $lon
        sender_balance_before   = $bal
        sender_balance_after    = [double]($bal - $amount)
        receiver_balance_before = [double](Get-Random -Min 10000 -Max 200000)
        receiver_balance_after  = [double](Get-Random -Min 10000 -Max 200000)
        device_id               = $device
        ip_address              = ""
        remarks                 = $remarks
        merchant_category       = ""
    }
}

# ------------------------------------------------------------------
# Helper: POST transaction and print one-line result
# ------------------------------------------------------------------
function Post-Tx($body, [string]$label = "") {
    try {
        $resp  = Invoke-RestMethod -Uri $URL -Method Post `
                 -Body ($body | ConvertTo-Json) -Headers $HDR
        $score = $resp.transaction_risk_score
        $rules = ($resp.rule_out.triggered_rules) -join ", "
        $level = $resp.risk_level
        $col   = if ($level -eq "Critical") { "Magenta" } `
            elseif ($level -eq "High")     { "Red"     } `
            elseif ($level -eq "Medium")   { "Yellow"  } `
            else                           { "Green"   }
        Write-Host ("  {0,-28} score={1,6}  level={2,-8}  rules=[{3}]" `
            -f $label, $score, $level, $rules) -ForegroundColor $col
        return $resp
    } catch {
        $s = $_.Exception.Response
        if ($s) {
            $r   = [System.IO.StreamReader]::new($s.GetResponseStream())
            $msg = "HTTP $($s.StatusCode.value__): $($r.ReadToEnd())"
        } else {
            $msg = $_.Exception.Message
        }
        Write-Host "  $label  ERROR: $msg" -ForegroundColor DarkRed
        return $null
    }
}

Write-Host ""
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host "  AML Rule Demo -- R01 through R15" -ForegroundColor Cyan
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host ""


# ------------------------------------------------------------------
# R01  high_value_transfer
#      condition : amount > 1,000,000
#      trigger   : single transaction over threshold
# ------------------------------------------------------------------
Write-Host "[R01] High-Value Transfer  (amount > 1,000,000)" -ForegroundColor Magenta

Post-Tx (Tx -id "R01_001" -sender "R01_SND" -receiver "R01_RCV" `
            -amount 1500000 -ts "2024-07-01T10:00:00Z") "R01_TRIGGER(1.5M)"

Post-Tx (Tx -id "R01_002" -sender "R01_SND_NO" -receiver "R01_RCV" `
            -amount 999999  -ts "2024-07-01T10:01:00Z") "R01_no_trigger(999K)"


# ------------------------------------------------------------------
# R02  structuring
#      condition : sub_threshold_flag AND tx_velocity_1h >= 3
#      sub_threshold = amount in [850,000 , 1,000,000)
#      tx_velocity_1h is PRE-EVENT -> need 3 prior txns in 1h window
#      -> send 4 sub-threshold txns; rule fires on the 4th
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R02] Structuring  (3 prior sub-threshold txns in 1h; fires on 4th)" -ForegroundColor Magenta

Post-Tx (Tx -id "R02_001" -sender "R02_SND" -receiver "R02_RX1" `
            -amount 899000 -ts "2024-07-02T09:00:00Z") "R02_seed1(pre-v=0)"
Post-Tx (Tx -id "R02_002" -sender "R02_SND" -receiver "R02_RX2" `
            -amount 875000 -ts "2024-07-02T09:15:00Z") "R02_seed2(pre-v=1)"
Post-Tx (Tx -id "R02_003" -sender "R02_SND" -receiver "R02_RX3" `
            -amount 950000 -ts "2024-07-02T09:30:00Z") "R02_seed3(pre-v=2)"
Post-Tx (Tx -id "R02_004" -sender "R02_SND" -receiver "R02_RX4" `
            -amount 920000 -ts "2024-07-02T09:45:00Z") "R02_TRIGGER(pre-v=3)"


# ------------------------------------------------------------------
# R03  velocity_spike
#      condition : tx_velocity_1h > 10
#      tx_velocity_1h is PRE-EVENT -> need 11 prior txns in 1h
#      -> send 12 txns; rule fires on the 12th
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R03] Velocity Spike  (11 prior txns in 1h; fires on 12th)" -ForegroundColor Magenta

for ($j = 0; $j -lt 12; $j++) {
    $min   = "{0:D2}" -f ($j * 4)
    $txid  = "R03_{0:D3}" -f ($j + 1)
    $rxacc = "R03_RX{0:D2}" -f ($j + 1)
    if ($j -eq 11) {
        $label = "R03_TRIGGER(pre-v=11)"
    } else {
        $label = "R03_seed$($j+1)(pre-v=$j)"
    }
    Post-Tx (Tx -id $txid -sender "R03_SND" -receiver $rxacc `
                -amount 30000 -ts "2024-07-03T08:${min}:00Z") $label
}


# ------------------------------------------------------------------
# R04  dormant_activation
#      condition : tx_gap_seconds > 15,552,000 AND amount > 100,000
#                  15,552,000 s = 180 days
#      gap = current_ts - last_tx_ts_ever  (never-trimmed timestamp)
#      Step 1: seed with 2023 timestamp -> sets last_tx_ts_ever
#      Step 2: send 2024 tx -> gap = 242 days > 180 days -> fires
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R04] Dormant Activation  (gap > 180 days + amount > 100K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R04_001" -sender "R04_SND" -receiver "R04_RX1" `
            -amount 5000 -ts "2023-11-01T10:00:00Z") "R04_seed(2023-11-01)"
Post-Tx (Tx -id "R04_002" -sender "R04_SND" -receiver "R04_RX2" `
            -amount 200000 -ts "2024-07-01T10:00:00Z") "R04_TRIGGER(gap=242d)"


# ------------------------------------------------------------------
# R05  impossible_travel
#      condition : impossible_travel_flag == True  (speed > 900 km/h)
#      Mumbai (19.07, 72.88) -> London (51.51, -0.13) in 30 min
#      ~7,200 km / 0.5 h = 14,400 km/h >> 900
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R05] Impossible Travel  (Mumbai -> London in 30 min)" -ForegroundColor Magenta

Post-Tx (Tx -id "R05_001" -sender "R05_SND" -receiver "R05_RX1" `
            -amount 10000 -ts "2024-07-04T10:00:00Z" `
            -lat 19.07 -lon 72.88) "R05_seed(Mumbai)"
Post-Tx (Tx -id "R05_002" -sender "R05_SND" -receiver "R05_RX2" `
            -amount 10000 -ts "2024-07-04T10:30:00Z" `
            -lat 51.51 -lon (-0.13)) "R05_TRIGGER(London,14400kmh)"


# ------------------------------------------------------------------
# R06  high_risk_jurisdiction
#      condition : high_risk_country_flag AND is_international == True
#      High-risk countries: AE, MU, CN, NG, PK, CH
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R06] High-Risk Jurisdiction  (IN -> PK, is_international=True)" -ForegroundColor Magenta

Post-Tx (Tx -id "R06_001" -sender "R06_SND" -receiver "R06_OFFSHR" `
            -amount 800000 -ts "2024-07-04T11:00:00Z" `
            -type "Wire" -srcCountry "IN" -dstCountry "PK" `
            -intl $true -kyc 1) "R06_TRIGGER(PK,intl)"

Post-Tx (Tx -id "R06_002" -sender "R06_SND_NO" -receiver "R06_DOM" `
            -amount 800000 -ts "2024-07-04T11:01:00Z" `
            -srcCountry "IN" -dstCountry "IN" `
            -intl $false) "R06_no_trigger(domestic)"


# ------------------------------------------------------------------
# R07  device_anomaly
#      condition : new_device_flag AND amount > 200,000
#      new_device_flag = device not seen before for this account
#      Per-run device ID ensures it is fresh on every execution
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R07] Device Anomaly  (new device + amount > 200K)" -ForegroundColor Magenta

$devR07 = "DEV_R07_${RUN}"

Post-Tx (Tx -id "R07_001" -sender "R07_SND" -receiver "R07_RX1" `
            -amount 350000 -ts "2024-07-04T12:00:00Z" `
            -device $devR07) "R07_TRIGGER(new_dev)"

Post-Tx (Tx -id "R07_002" -sender "R07_SND" -receiver "R07_RX2" `
            -amount 350000 -ts "2024-07-04T12:05:00Z" `
            -device $devR07) "R07_no_trigger(same_dev)"


# ------------------------------------------------------------------
# R08  shared_device
#      condition : shared_device_count > 5
#      shared_device_count is PRE-EVENT -> need 6 accounts already
#      ingested before the 7th evaluation reaches count = 6 > 5
#      -> send 7 accounts; rule fires on the 7th
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R08] Shared Device  (6 prior accounts on same device; fires on 7th)" -ForegroundColor Magenta

$devR08 = "DEV_SHARED_R08_${RUN}"
1..7 | ForEach-Object {
    $i    = $_
    $txid = "R08_{0:D3}" -f $i
    $acct = "R08_MULE{0:D2}" -f $i
    if ($i -eq 7) {
        $label = "R08_TRIGGER(pre-cnt=6)"
    } else {
        $label = "R08_seed${i}(pre-cnt=$($i-1))"
    }
    Post-Tx (Tx -id $txid -sender $acct -receiver "R08_HUB" `
                -amount 50000 -ts "2024-07-04T13:0${i}:00Z" `
                -device $devR08) $label
}


# ------------------------------------------------------------------
# R09  excessive_beneficiaries
#      condition : beneficiary_count_7d > 20
#      beneficiary_count_7d is PRE-EVENT -> need 21 unique receivers
#      already stored before the 22nd evaluation reaches 21 > 20
#      -> send 22 txns to 22 unique receivers; rule fires on the 22nd
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R09] Excessive Beneficiaries  (21 prior unique receivers; fires on 22nd)" -ForegroundColor Magenta

1..22 | ForEach-Object {
    $i    = $_
    $txid = "R09_{0:D3}" -f $i
    $recv = "R09_RECV{0:D2}" -f $i
    $day  = "{0:D2}" -f [int]([math]::Floor(($i - 1) / 4) + 1)
    $hr   = "{0:D2}" -f [int]((($i - 1) % 4) * 6)
    if ($i -eq 22) {
        $label = "R09_TRIGGER(pre-cnt=21)"
    } else {
        $label = "R09_seed${i}(pre-cnt=$($i-1))"
    }
    Post-Tx (Tx -id $txid -sender "R09_SND" -receiver $recv `
                -amount 5000 -ts "2024-07-${day}T${hr}:00:00Z") $label
}


# ------------------------------------------------------------------
# R10  cycle_closure
#      condition : edge_creates_cycle == True
#      Build A->B->C->A; the 3rd edge closes the cycle
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R10] Cycle Closure  (A->B->C->A circular laundering)" -ForegroundColor Magenta

Post-Tx (Tx -id "R10_001" -sender "R10_A" -receiver "R10_B" `
            -amount 500000 -ts "2024-07-04T14:00:00Z") "R10_A->B(no_cycle)"
Post-Tx (Tx -id "R10_002" -sender "R10_B" -receiver "R10_C" `
            -amount 490000 -ts "2024-07-04T14:30:00Z") "R10_B->C(no_cycle)"
Post-Tx (Tx -id "R10_003" -sender "R10_C" -receiver "R10_A" `
            -amount 480000 -ts "2024-07-04T15:00:00Z") "R10_TRIGGER(C->A,cycle)"


# ------------------------------------------------------------------
# R11  round_amount_structuring
#      condition : round_amount_flag AND tx_velocity_24h > 5
#      tx_velocity_24h is PRE-EVENT -> need 6 prior round-amount txns
#      -> send 7 round-amount txns; rule fires on the 7th
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R11] Round Amount Structuring  (6 prior round txns in 24h; fires on 7th)" -ForegroundColor Magenta

$r11amounts = @(300000, 500000, 200000, 100000, 400000, 600000, 700000)
for ($j = 0; $j -lt 7; $j++) {
    $hr    = "{0:D2}" -f ($j * 3)
    $txid  = "R11_{0:D3}" -f ($j + 1)
    $rxacc = "R11_RX{0:D2}" -f ($j + 1)
    if ($j -eq 6) {
        $label = "R11_TRIGGER(pre-v=6)"
    } else {
        $label = "R11_seed$($j+1)(pre-v=$j)"
    }
    Post-Tx (Tx -id $txid -sender "R11_SND" -receiver $rxacc `
                -amount $r11amounts[$j] -ts "2024-07-04T${hr}:00:00Z") $label
}


# ------------------------------------------------------------------
# R12  kyc_mismatch
#      condition : kyc_level == 0 AND amount > 500,000
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R12] KYC Mismatch  (kyc_level=0 + amount > 500K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R12_001" -sender "R12_SND"    -receiver "R12_RX1" `
            -amount 750000 -ts "2024-07-04T16:00:00Z" -kyc 0) "R12_TRIGGER(kyc=0)"

Post-Tx (Tx -id "R12_002" -sender "R12_SND_NO" -receiver "R12_RX1" `
            -amount 750000 -ts "2024-07-04T16:01:00Z" -kyc 1) "R12_no_trigger(kyc=1)"


# ------------------------------------------------------------------
# R13  benford_anomaly
#      condition : benford_chi2_score > 3.84
#      chi2 uses st.all_amounts (PRE-EVENT, min 5 entries required)
#      All 9X,XXX amounts start with digit 9 -> extreme chi2
#      Fires from the 6th transaction onward (5 prior amounts present)
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R13] Benford Anomaly  (all amounts start with 9; fires from tx6)" -ForegroundColor Magenta

$r13amounts = @(90000, 91000, 92000, 93000, 94000, 95000, 96000, 97000,
                98000, 99000, 90500, 91500, 92500, 93500, 94500)
for ($j = 0; $j -lt 15; $j++) {
    $hr    = "{0:D2}" -f ($j + 1)
    $txid  = "R13_{0:D3}" -f ($j + 1)
    $rxacc = "R13_RX{0:D2}" -f ($j + 1)
    if ($j -ge 5) {
        $label = "R13_TRIGGER_tx$($j+1)(${j}prior)"
    } else {
        $label = "R13_seed$($j+1)(${j}prior)"
    }
    Post-Tx (Tx -id $txid -sender "R13_SND" -receiver $rxacc `
                -amount $r13amounts[$j] -ts "2024-07-04T${hr}:00:00Z") $label
}


# ------------------------------------------------------------------
# R14  fan_in_collector
#      condition : receiver_in_degree_unique >= 3
#      receiver_in_degree_unique is PRE-EVENT graph count
#      -> 3 senders must be ingested before the 4th evaluation
#      -> send 4 senders to 1 collector; fires on the 4th
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R14] Fan-In Collector  (3 prior unique senders; fires on 4th)" -ForegroundColor Magenta

1..4 | ForEach-Object {
    $i    = $_
    $txid = "R14_{0:D3}" -f $i
    $src  = "R14_SRC{0:D2}" -f $i
    if ($i -eq 4) {
        $label = "R14_TRIGGER(pre-indeg=3)"
    } else {
        $label = "R14_seed${i}(pre-indeg=$($i-1))"
    }
    Post-Tx (Tx -id $txid -sender $src -receiver "R14_COLLECT" `
                -amount 200000 -ts "2024-07-04T1${i}:00:00Z") $label
}


# ------------------------------------------------------------------
# R15  passthrough_layering
#      condition : sender_in_degree >= 1 AND amount >= 500,000
#      sender_in_degree is PRE-EVENT graph count
#      Step 1: seed R15_MID as a receiver (graph in_degree becomes 1)
#      Step 2: R15_MID sends >= 500K forward -> sender_in_degree=1 -> fires
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[R15] Pass-Through Layering  (receive first, then forward >= 500K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R15_001" -sender "R15_SRC" -receiver "R15_MID" `
            -amount 800000 -ts "2024-07-04T17:00:00Z") "R15_seed(MID_receives)"
Post-Tx (Tx -id "R15_002" -sender "R15_MID" -receiver "R15_DST" `
            -amount 750000 -ts "2024-07-04T17:30:00Z") "R15_TRIGGER(MID_forwards)"


# ------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host "  All 15 rule demos sent.  Run ID: $RUN" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "  Rule  | Fires on    | Key condition (PRE-event count)" -ForegroundColor Yellow
Write-Host "  ------+-------------+---------------------------------" -ForegroundColor Yellow
Write-Host "  R01   | tx1         | amount > 1,000,000" -ForegroundColor Yellow
Write-Host "  R02   | tx4         | v1h >= 3  (3 prior sub-threshold txns in 1h)" -ForegroundColor Yellow
Write-Host "  R03   | tx12        | v1h > 10  (11 prior txns in 1h)" -ForegroundColor Yellow
Write-Host "  R04   | tx2         | gap > 15.5M s  (242 day seed gap)" -ForegroundColor Yellow
Write-Host "  R05   | London tx   | impossible_travel_flag = True" -ForegroundColor Yellow
Write-Host "  R06   | PK wire     | high_risk_country AND is_international" -ForegroundColor Yellow
Write-Host "  R07   | tx1         | new_device_flag AND amount > 200K" -ForegroundColor Yellow
Write-Host "  R08   | tx7         | shared_device_count > 5  (6 prior accounts)" -ForegroundColor Yellow
Write-Host "  R09   | tx22        | beneficiary_7d > 20  (21 prior receivers)" -ForegroundColor Yellow
Write-Host "  R10   | C->A tx     | edge_creates_cycle = True" -ForegroundColor Yellow
Write-Host "  R11   | tx7         | v24h > 5  (6 prior round-amount txns in 24h)" -ForegroundColor Yellow
Write-Host "  R12   | tx1         | kyc_level == 0 AND amount > 500K" -ForegroundColor Yellow
Write-Host "  R13   | tx6+        | benford_chi2 > 3.84  (5 prior amounts)" -ForegroundColor Yellow
Write-Host "  R14   | tx4         | receiver_in_degree >= 3  (3 prior senders)" -ForegroundColor Yellow
Write-Host "  R15   | MID-fwd tx  | sender_in_degree >= 1 AND amount >= 500K" -ForegroundColor Yellow
Write-Host ("=" * 72) -ForegroundColor Cyan
