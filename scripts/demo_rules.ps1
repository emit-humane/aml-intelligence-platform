# =============================================================
# demo_rules.ps1  —  One fraud scenario per rule, R01-R15
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\demo_rules.ps1
# =============================================================

$URL  = "https://aml-intelligence-platform.onrender.com/stream/transaction"
$HDR  = @{ "Content-Type" = "application/json" }
$BANKS = @("SBI","HDFC","ICICI","AXIS","KOTAK","PNB")

function Tx {
    param(
        [string]$id, [string]$sender, [string]$receiver,
        [double]$amount, [string]$ts,
        [string]$type="NEFT", [string]$channel="Web",
        [string]$srcCountry="IN", [string]$dstCountry="IN",
        [bool]$intl=$false, [int]$kyc=2,
        [double]$lat=20.0, [double]$lon=78.0,
        [string]$device="", [string]$remarks=""
    )
    $bal = [double](Get-Random -Min 500000 -Max 5000000)
    return @{
        transaction_id          = $id
        sender_account          = $sender
        receiver_account        = $receiver
        sender_name             = "Demo_$sender"
        receiver_name           = "Demo_$receiver"
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

function Post-Tx($body, [string]$label="") {
    try {
        $resp = Invoke-RestMethod -Uri $URL -Method Post `
                -Body ($body | ConvertTo-Json) -Headers $HDR
        $score   = $resp.transaction_risk_score
        $rules   = $resp.rule_out.triggered_rules -join ", "
        $level   = $resp.risk_level
        $col     = switch ($level) {
            "Critical" { "Magenta" }; "High" { "Red" }
            "Medium"   { "Yellow"  }; default { "Green" }
        }
        Write-Host ("  {0,-18} score={1,6}  level={2,-8}  rules=[{3}]" `
            -f $label, $score, $level, $rules) -ForegroundColor $col
        return $resp
    } catch {
        $s = $_.Exception.Response
        $msg = if ($s) {
            $r = [System.IO.StreamReader]::new($s.GetResponseStream())
            "HTTP $($s.StatusCode.value__): $($r.ReadToEnd())"
        } else { $_.Exception.Message }
        Write-Host "  $label  ERROR: $msg" -ForegroundColor DarkRed
        return $null
    }
}

Write-Host "`n$('='*65)" -ForegroundColor Cyan
Write-Host "  AML Rule Demo  —  R01 through R15" -ForegroundColor Cyan
Write-Host "$('='*65)`n" -ForegroundColor Cyan


# ─────────────────────────────────────────────────────────────
# R01  high_value_transfer   condition: amount > 1_000_000
# ─────────────────────────────────────────────────────────────
Write-Host "[R01] High-Value Transfer  (amount > 1,000,000)" -ForegroundColor Magenta

Post-Tx (Tx -id "R01_001" -sender "ACC_R01_01" -receiver "ACC_R01_02" `
            -amount 1500000 -ts "2024-07-01T10:00:00Z") "R01_trigger"

Post-Tx (Tx -id "R01_002" -sender "ACC_R01_NO" -receiver "ACC_R01_02" `
            -amount 999999 -ts "2024-07-01T10:01:00Z") "R01_no_trigger"


# ─────────────────────────────────────────────────────────────
# R02  structuring   condition: sub_threshold_flag AND tx_velocity_1h >= 3
#      sub_threshold = amount in [850_000, 1_000_000)
#      Send 3 sub-threshold txns from same account within 1 hour
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R02] Structuring  (3x sub-threshold in 1 hour)" -ForegroundColor Magenta

Post-Tx (Tx -id "R02_001" -sender "ACC_R02_01" -receiver "ACC_R02_RX" `
            -amount 899000 -ts "2024-07-02T09:00:00Z") "R02_setup_1"
Post-Tx (Tx -id "R02_002" -sender "ACC_R02_01" -receiver "ACC_R02_RY" `
            -amount 875000 -ts "2024-07-02T09:15:00Z") "R02_setup_2"
Post-Tx (Tx -id "R02_003" -sender "ACC_R02_01" -receiver "ACC_R02_RZ" `
            -amount 950000 -ts "2024-07-02T09:30:00Z") "R02_trigger"


# ─────────────────────────────────────────────────────────────
# R03  velocity_spike   condition: tx_velocity_1h > 10 OR tx_velocity_24h > 50
#      Send 11 transactions within 1 hour from same account
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R03] Velocity Spike  (11 txns in 1 hour)" -ForegroundColor Magenta

$r03recvs = @("RX01","RX02","RX03","RX04","RX05","RX06","RX07","RX08","RX09","RX10","RX11")
for ($j = 0; $j -lt 11; $j++) {
    $min   = "{0:D2}" -f ($j * 5)
    $txid  = "R03_{0:D3}" -f ($j + 1)
    $label = if ($j -eq 10) { "R03_trigger(11)" } else { "R03_setup_$($j+1)" }
    Post-Tx (Tx -id $txid -sender "ACC_R03_01" `
                -receiver ("ACC_R03_$($r03recvs[$j])") `
                -amount 30000 -ts "2024-07-03T08:${min}:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R04  dormant_activation  condition: tx_gap_seconds > 15_552_000 AND amount > 100_000
#      15_552_000s = 180 days
#      Step 1: send one transaction with old timestamp (>180 days ago)
#      Step 2: send a transaction NOW — gap will be >180 days
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R04] Dormant Activation  (gap > 180 days, amount > 100K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R04_001" -sender "ACC_R04_01" -receiver "ACC_R04_RX" `
            -amount 5000 -ts "2023-11-01T10:00:00Z") "R04_dormant_seed"
Post-Tx (Tx -id "R04_002" -sender "ACC_R04_01" -receiver "ACC_R04_RY" `
            -amount 200000 -ts "2024-07-01T10:00:00Z") "R04_trigger(242d)"


# ─────────────────────────────────────────────────────────────
# R05  impossible_travel   condition: impossible_travel_flag == True
#      Speed between consecutive transactions > 900 km/h
#      Mumbai (19.07, 72.88) -> London (51.51, -0.13) in 30 min
#      Distance ~7200 km, speed ~14400 km/h  >> 900
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R05] Impossible Travel  (Mumbai -> London in 30 min)" -ForegroundColor Magenta

Post-Tx (Tx -id "R05_001" -sender "ACC_R05_01" -receiver "ACC_R05_RX" `
            -amount 10000 -ts "2024-07-01T10:00:00Z" `
            -lat 19.07 -lon 72.88) "R05_Mumbai"
Post-Tx (Tx -id "R05_002" -sender "ACC_R05_01" -receiver "ACC_R05_RY" `
            -amount 10000 -ts "2024-07-01T10:30:00Z" `
            -lat 51.51 -lon (-0.13)) "R05_trigger(London)"


# ─────────────────────────────────────────────────────────────
# R06  high_risk_jurisdiction
#      condition: high_risk_country_flag AND is_international == True
#      High-risk countries: AE, MU, CN, NG, PK, CH
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R06] High-Risk Jurisdiction  (IN -> PK, is_international)" -ForegroundColor Magenta

Post-Tx (Tx -id "R06_001" -sender "ACC_R06_01" -receiver "OFFSHR_PK_01" `
            -amount 800000 -ts "2024-07-01T11:00:00Z" `
            -type "Wire" -srcCountry "IN" -dstCountry "PK" `
            -intl $true -kyc 1) "R06_trigger(PK)"

# No-trigger: same amount but domestic
Post-Tx (Tx -id "R06_002" -sender "ACC_R06_NO" -receiver "ACC_R06_RX" `
            -amount 800000 -ts "2024-07-01T11:01:00Z" `
            -srcCountry "IN" -dstCountry "IN" -intl $false) "R06_no_trigger"


# ─────────────────────────────────────────────────────────────
# R07  device_anomaly   condition: new_device_flag AND amount > 200_000
#      new_device_flag = True when account uses a device_id it has never used before
#      First-ever transaction from a fresh account with a real device_id triggers this
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R07] Device Anomaly  (new device + amount > 200K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R07_001" -sender "ACC_R07_01" -receiver "ACC_R07_RX" `
            -amount 350000 -ts "2024-07-01T12:00:00Z" `
            -device "DEVICE_R07_NEW_001") "R07_trigger"

# Second tx from same account, same device — no longer new
Post-Tx (Tx -id "R07_002" -sender "ACC_R07_01" -receiver "ACC_R07_RY" `
            -amount 350000 -ts "2024-07-01T12:05:00Z" `
            -device "DEVICE_R07_NEW_001") "R07_no_trigger(same_dev)"


# ─────────────────────────────────────────────────────────────
# R08  shared_device   condition: shared_device_count > 5
#      Same physical device used by more than 5 distinct accounts
#      Send transactions from 6 different accounts using the SAME device_id
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R08] Shared Device  (1 device used by 6+ accounts)" -ForegroundColor Magenta

$sharedDev = "SHARED_DEV_R08_MULE"
1..6 | ForEach-Object {
    $i     = $_
    $txid  = "R08_{0:D3}" -f $i
    $acct  = "ACC_R08_{0:D2}" -f $i
    $label = if ($i -ge 6) { "R08_trigger(acct$i)" } else { "R08_setup(acct$i)" }
    Post-Tx (Tx -id $txid `
                -sender $acct `
                -receiver "ACC_R08_RX" `
                -amount 50000 -ts "2024-07-01T13:0${i}:00Z" `
                -device $sharedDev) $label
}


# ─────────────────────────────────────────────────────────────
# R09  excessive_beneficiaries   condition: beneficiary_count_7d > 20
#      One sender pays 21 distinct receivers within 7 days
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R09] Excessive Beneficiaries  (21 unique receivers in 7 days)" -ForegroundColor Magenta

1..21 | ForEach-Object {
    $i    = $_
    $txid = "R09_{0:D3}" -f $i
    $recv = "ACC_R09_RECV_{0:D2}" -f $i
    $day  = "{0:D2}" -f ([math]::Floor(($i - 1) / 4) + 1)
    $hr   = "{0:D2}" -f ((($i - 1) % 4) * 6)
    $label = if ($i -eq 21) { "R09_trigger(recv21)" } else { "R09_setup(recv$i)" }
    Post-Tx (Tx -id $txid `
                -sender "ACC_R09_01" `
                -receiver $recv `
                -amount 5000 `
                -ts "2024-07-${day}T${hr}:00:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R10  cycle_closure   condition: edge_creates_cycle == True
#      Build cycle: A -> B -> C -> A
#      Third transaction (C->A) closes the cycle
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R10] Cycle Closure  (A->B->C->A circular laundering)" -ForegroundColor Magenta

Post-Tx (Tx -id "R10_001" -sender "ACC_R10_A" -receiver "ACC_R10_B" `
            -amount 500000 -ts "2024-07-01T14:00:00Z") "R10_A->B"
Post-Tx (Tx -id "R10_002" -sender "ACC_R10_B" -receiver "ACC_R10_C" `
            -amount 490000 -ts "2024-07-01T14:30:00Z") "R10_B->C"
Post-Tx (Tx -id "R10_003" -sender "ACC_R10_C" -receiver "ACC_R10_A" `
            -amount 480000 -ts "2024-07-01T15:00:00Z") "R10_trigger(C->A)"


# ─────────────────────────────────────────────────────────────
# R11  round_amount_structuring
#      condition: round_amount_flag AND tx_velocity_24h > 5
#      round_amount_flag = amount is multiple of 100_000
#      Send 6 round-amount transactions in the same 24h window
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R11] Round Amount Structuring  (6x round amounts in 24h)" -ForegroundColor Magenta

$r11amounts = @(1000000, 500000, 200000, 100000, 300000, 400000)
for ($j = 0; $j -lt 6; $j++) {
    $hr    = "{0:D2}" -f ($j * 3)
    $txid  = "R11_{0:D3}" -f ($j + 1)
    $rxacc = "ACC_R11_RX_{0:D2}" -f ($j + 1)
    $label = if ($j -eq 5) { "R11_trigger(6th)" } else { "R11_setup_$($j+1)" }
    Post-Tx (Tx -id $txid `
                -sender "ACC_R11_01" `
                -receiver $rxacc `
                -amount $r11amounts[$j] `
                -ts "2024-07-01T${hr}:00:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R12  kyc_mismatch   condition: kyc_level == 0 AND amount > 500_000
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R12] KYC Mismatch  (kyc_level=0, amount > 500K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R12_001" -sender "ACC_R12_01" -receiver "ACC_R12_RX" `
            -amount 750000 -ts "2024-07-01T16:00:00Z" -kyc 0) "R12_trigger"

Post-Tx (Tx -id "R12_002" -sender "ACC_R12_NO" -receiver "ACC_R12_RX" `
            -amount 750000 -ts "2024-07-01T16:01:00Z" -kyc 1) "R12_no_trigger(kyc=1)"


# ─────────────────────────────────────────────────────────────
# R13  benford_anomaly   condition: benford_chi2_score > 3.84
#      chi2 uses account's transaction history (min 9 txns)
#      All amounts start with digit 9 -> extreme Benford violation
#      Benford expects ~4.6% for digit 9; 100% gives chi2 >> 3.84
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R13] Benford Anomaly  (15 txns, all amounts start with 9)" -ForegroundColor Magenta

$r13base = @(90000,91000,92000,93000,94000,95000,96000,97000,98000,99000,
             90500,91500,92500,93500,94500)
for ($j = 0; $j -lt 15; $j++) {
    $hr    = "{0:D2}" -f ($j + 1)
    $txid  = "R13_{0:D3}" -f ($j + 1)
    $rxacc = "ACC_R13_RX_{0:D2}" -f ($j + 1)
    $label = if ($j -ge 8) { "R13_trigger(tx$($j+1))" } else { "R13_setup_$($j+1)" }
    Post-Tx (Tx -id $txid `
                -sender "ACC_R13_01" `
                -receiver $rxacc `
                -amount $r13base[$j] `
                -ts "2024-07-01T${hr}:00:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R14  fan_in_collector
#      condition: receiver_in_degree_unique >= 3
#      3 distinct senders all paying the same collector account
#      R14 fires on (and after) the 3rd unique sender
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R14] Fan-In Collector  (3 unique senders -> 1 collector)" -ForegroundColor Magenta

1..3 | ForEach-Object {
    $i     = $_
    $txid  = "R14_{0:D3}" -f $i
    $src   = "ACC_R14_SRC_{0:D2}" -f $i
    $label = if ($i -eq 3) { "R14_trigger(3rd_sender)" } else { "R14_setup(sender$i)" }
    Post-Tx (Tx -id $txid `
                -sender $src `
                -receiver "ACC_R14_COLLECT" `
                -amount 200000 `
                -ts "2024-07-01T1${i}:00:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R15  passthrough_layering
#      condition: sender_in_degree >= 1 AND amount >= 500_000
#      Step 1: seed ACC_R15_MID as a receiver (gives it in_degree=1)
#      Step 2: ACC_R15_MID sends >= 500K onward (triggers R15)
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R15] Pass-Through Layering  (receive then forward >= 500K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R15_001" -sender "ACC_R15_SRC" -receiver "ACC_R15_MID" `
            -amount 800000 -ts "2024-07-01T17:00:00Z") "R15_seed(MID receives)"
Post-Tx (Tx -id "R15_002" -sender "ACC_R15_MID" -receiver "ACC_R15_DST" `
            -amount 750000 -ts "2024-07-01T17:30:00Z") "R15_trigger(MID forwards)"


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
Write-Host "`n$('='*65)" -ForegroundColor Cyan
Write-Host "  All 15 rule demos sent. Check scores above." -ForegroundColor Cyan
Write-Host "  Rules that need state buildup (R02/R03/R08/R09/R11/R13):" -ForegroundColor Yellow
Write-Host "    -> Only the LAST transaction in each group shows the rule firing." -ForegroundColor Yellow
Write-Host "$('='*65)" -ForegroundColor Cyan
