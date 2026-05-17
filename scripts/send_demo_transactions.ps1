# ============================================================
# send_demo_transactions.ps1
# Posts 80 normal + 20 fraud transactions to the live API
# Usage:  powershell -ExecutionPolicy Bypass -File .\scripts\send_demo_transactions.ps1
# ============================================================

$URL   = "https://aml-intelligence-platform.onrender.com/stream/transaction"
$HDR   = @{ "Content-Type" = "application/json" }
$BANKS = @("SBI","HDFC","ICICI","AXIS","KOTAK","PNB")
$ok      = 0
$fail    = 0
$flagged = 0

function Send-Tx($body) {
    try {
        $resp = Invoke-RestMethod -Uri $URL -Method Post -Body ($body | ConvertTo-Json) -Headers $HDR
        return $resp
    } catch {
        # Uncomment next 3 lines to print full server error:
        # $s = $_.Exception.Response
        # $r = [System.IO.StreamReader]::new($s.GetResponseStream())
        # Write-Host "    SERVER: $($r.ReadToEnd())" -ForegroundColor DarkYellow
        return $null
    }
}

# Builds a fully-populated TransactionEvent hashtable.
# All required Pydantic fields included — avoids 422/500 from missing fields.
function New-Base {
    param(
        [string]$txId,
        [string]$sender,
        [string]$receiver,
        [double]$amount,
        [string]$ts,
        [string]$type,
        [string]$channel,
        [string]$srcCountry = "IN",
        [string]$dstCountry = "IN",
        [bool]  $intl       = $false,
        [int]   $kyc        = 2
    )
    $bal  = Get-Random -Min 200000 -Max 2000000
    $rbal = Get-Random -Min 50000  -Max 1000000
    return @{
        transaction_id          = $txId
        sender_account          = $sender
        receiver_account        = $receiver
        sender_name             = "User_$sender"
        receiver_name           = "User_$receiver"
        amount                  = $amount
        currency                = "INR"
        timestamp               = $ts
        transaction_type        = $type
        payment_channel         = $channel
        sender_country          = $srcCountry
        receiver_country        = $dstCountry
        sender_bank             = $BANKS[(Get-Random -Min 0 -Max 6)]
        receiver_bank           = $BANKS[(Get-Random -Min 0 -Max 6)]
        transaction_status      = "Success"
        is_international        = $intl
        kyc_level               = $kyc
        geo_latitude            = [math]::Round(8  + (Get-Random -Min 0 -Max 2700) / 100.0, 4)
        geo_longitude           = [math]::Round(68 + (Get-Random -Min 0 -Max 2900) / 100.0, 4)
        sender_balance_before   = [double]$bal
        sender_balance_after    = [double]($bal - $amount)
        receiver_balance_before = [double]$rbal
        receiver_balance_after  = [double]($rbal + $amount)
        device_id               = ""
        ip_address              = ""
        remarks                 = ""
        merchant_category       = ""
    }
}

# ── 80 NORMAL ─────────────────────────────────────────────────────────────────
Write-Host "`n=== Sending 80 NORMAL transactions ===" -ForegroundColor Cyan

1..80 | ForEach-Object {
    $i    = $_
    $s    = "ACC{0:D6}" -f (Get-Random -Min 1 -Max 3000)
    $r    = "ACC{0:D6}" -f (Get-Random -Min 1 -Max 3000)
    $amt  = Get-Random -Min 500 -Max 49999
    $hr   = "{0:D2}" -f (Get-Random -Min 8 -Max 20)
    $mn   = "{0:D2}" -f (Get-Random -Min 0 -Max 59)
    $type = @("NEFT","IMPS","UPI","RTGS")[(Get-Random -Min 0 -Max 4)]
    $ch   = @("Web","Mobile","Branch","ATM")[(Get-Random -Min 0 -Max 4)]

    $tx   = New-Base -txId    ("NRM{0:D5}" -f $i) `
                     -sender   $s `
                     -receiver $r `
                     -amount   $amt `
                     -ts       "2024-06-15T${hr}:${mn}:00Z" `
                     -type     $type `
                     -channel  $ch

    $resp = Send-Tx $tx
    if ($resp) {
        $ok++
        if ($resp.transaction_risk_score -ge 31) { $flagged++ }
        $col = if ($resp.transaction_risk_score -ge 31) { "Yellow" } else { "Green" }
        $ruleNames = $resp.rule_out.triggered_rules -join ", "
        Write-Host ("  N{0:D3}  score={1,3}  patterns={2}  rules=[{3}]" -f `
            $i, $resp.transaction_risk_score, $resp.triggered_patterns.Count, $ruleNames) `
            -ForegroundColor $col
    } else {
        $fail++
        Write-Host "  N$i  FAILED" -ForegroundColor Red
    }
}

# ── 20 FRAUD (4 typologies × 5) ───────────────────────────────────────────────
Write-Host "`n=== Sending 20 FRAUD transactions (4 typologies x5) ===" -ForegroundColor Cyan

# ── [1/4] STRUCTURING ─ 5 × ~499K, same sender, 5 consecutive hours ──────────
Write-Host "`n[1/4] Structuring  (ACC_STRUCT_01 sends 5x ~499K in 5 hrs)" -ForegroundColor Magenta
1..5 | ForEach-Object {
    $i   = $_
    $amt = 499000 + (Get-Random -Min 0 -Max 999)
    $rec = "ACC{0:D6}" -f (Get-Random -Min 4000 -Max 4999)
    $hr  = "{0:D2}" -f (6 + $i)          # 07:00 → 11:00 — build velocity

    $tx  = New-Base -txId    ("STRUCT{0:D3}" -f $i) `
                    -sender   "ACC_STRUCT_01" `
                    -receiver $rec `
                    -amount   $amt `
                    -ts       "2024-06-15T${hr}:15:00Z" `
                    -type     "NEFT" `
                    -channel  "Web" `
                    -kyc      1

    $resp = Send-Tx $tx
    if ($resp) {
        $ok++; $flagged++
        $ruleNames = $resp.rule_out.triggered_rules -join ", "
        Write-Host ("  STRUCT{0:D3}  amt={1}  score={2,3}  rules=[{3}]" -f `
            $i, $amt, $resp.transaction_risk_score, $ruleNames) -ForegroundColor Red
    } else {
        $fail++
        Write-Host "  STRUCT$i  FAILED" -ForegroundColor DarkRed
    }
}

# ── [2/4] LARGE WIRE ─ 5 × >5M INR, cross-border, high-risk country ──────────
Write-Host "`n[2/4] Large Wire   (5x >5M INR to high-risk countries)" -ForegroundColor Magenta
$hiRisk = @("PK","AE","NG","KE","CN")
1..5 | ForEach-Object {
    $i   = $_
    $amt = 5000000 + (Get-Random -Min 0 -Max 4000000)
    $dst = $hiRisk[$i - 1]
    $hr  = "{0:D2}" -f (10 + $i)         # 11:00 → 15:00

    $tx  = New-Base -txId    ("WIRE{0:D3}" -f $i) `
                    -sender   ("ACC_WIRE_{0:D2}" -f $i) `
                    -receiver ("OFFSHR_{0:D2}"   -f $i) `
                    -amount   $amt `
                    -ts       "2024-06-15T${hr}:30:00Z" `
                    -type     "Wire" `
                    -channel  "Web" `
                    -srcCountry "IN" `
                    -dstCountry $dst `
                    -intl     $true `
                    -kyc      1
    $tx.sender_bank   = "AXIS"
    $tx.receiver_bank = "FOREIGN_BANK"

    $resp = Send-Tx $tx
    if ($resp) {
        $ok++; $flagged++
        $ruleNames = $resp.rule_out.triggered_rules -join ", "
        Write-Host ("  WIRE{0:D3}  amt={1}  dst={2}  score={3,3}  rules=[{4}]" -f `
            $i, $amt, $dst, $resp.transaction_risk_score, $ruleNames) -ForegroundColor Red
    } else {
        $fail++
        Write-Host "  WIRE$i  FAILED" -ForegroundColor DarkRed
    }
}

# ── [3/4] FAN-IN ─ 5 different senders all funnelling into one collector ──────
Write-Host "`n[3/4] Fan-In       (5 senders -> ACC_COLLECT_01)" -ForegroundColor Magenta
1..5 | ForEach-Object {
    $i   = $_
    $amt = 150000 + (Get-Random -Min 0 -Max 99999)
    $mn  = "{0:D2}" -f ($i * 10)

    $tx  = New-Base -txId    ("FANIN{0:D3}" -f $i) `
                    -sender   ("ACC_FAN_{0:D2}" -f $i) `
                    -receiver "ACC_COLLECT_01" `
                    -amount   $amt `
                    -ts       "2024-06-15T14:${mn}:00Z" `
                    -type     "RTGS" `
                    -channel  "Web" `
                    -kyc      1
    $tx.receiver_bank = "KOTAK"

    $resp = Send-Tx $tx
    if ($resp) {
        $ok++; $flagged++
        $ruleNames = $resp.rule_out.triggered_rules -join ", "
        Write-Host ("  FANIN{0:D3}  amt={1}  score={2,3}  rules=[{3}]" -f `
            $i, $amt, $resp.transaction_risk_score, $ruleNames) -ForegroundColor Red
    } else {
        $fail++
        Write-Host "  FANIN$i  FAILED" -ForegroundColor DarkRed
    }
}

# ── [4/4] LAYERING CHAIN ─ A→B→C→D→E with round 2M INR amounts ───────────────
Write-Host "`n[4/4] Layering Chain  (A->B->C->D->E, round 2M amounts)" -ForegroundColor Magenta
$chain = @("ACC_CH_A","ACC_CH_B","ACC_CH_C","ACC_CH_D","ACC_CH_E","ACC_CH_EXIT")
0..4 | ForEach-Object {
    $i   = $_
    $src = $chain[$i]
    $dst = $chain[$i + 1]
    $amt = 2000000 - ($i * 50000)
    $mn  = "{0:D2}" -f ($i * 10)

    $tx  = New-Base -txId    ("CHAIN{0:D3}" -f ($i + 1)) `
                    -sender   $src `
                    -receiver $dst `
                    -amount   $amt `
                    -ts       "2024-06-15T15:${mn}:00Z" `
                    -type     "NEFT" `
                    -channel  "Web" `
                    -kyc      0
    $tx.sender_bank   = "ICICI"
    $tx.receiver_bank = "HDFC"

    $resp = Send-Tx $tx
    if ($resp) {
        $ok++; $flagged++
        $ruleNames = $resp.rule_out.triggered_rules -join ", "
        Write-Host ("  CHAIN{0:D3}  {1}->{2}  amt={3}  score={4,3}  rules=[{5}]" -f `
            ($i + 1), $src, $dst, $amt, $resp.transaction_risk_score, $ruleNames) -ForegroundColor Red
    } else {
        $fail++
        Write-Host "  CHAIN$($i + 1)  FAILED" -ForegroundColor DarkRed
    }
}

# ── SUMMARY ───────────────────────────────────────────────────────────────────
Write-Host "`n$("=" * 57)" -ForegroundColor Cyan
Write-Host ("DONE   sent={0}   failed={1}   flagged={2}/{0}" -f $ok, $fail, $flagged) -ForegroundColor White
$normalFlagged = $flagged - ($ok - 80)
Write-Host ("       normals flagged ~{0}/80    fraud flagged ~{1}/20" -f $normalFlagged, ($ok - 80)) -ForegroundColor White
