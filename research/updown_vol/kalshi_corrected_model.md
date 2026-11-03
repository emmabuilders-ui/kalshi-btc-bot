# Kalshi-corrected chance for hourly range bands (fitted 2026-11-03 on Dec-Aug, NOT used by any bot)

L = ln(mid / (1 - mid)), mid = (YES bid + YES ask)/2 ; g = ln(minutes left - 0.5)

    z = +0.23836 +1.30819*L -0.04436*g -0.09270*L*g -0.05855*L^2 +0.00607*L^2*g
    P(YES) = 1 / (1 + exp(-z))

Brier Sep-Oct: Kalshi 0.05233, this 0.05222 (table version 0.05224). Trading on it (YES at ask or NO at 1-bid when
corrected chance >= price + 2c, first minute per band) was about break-even in Sep-Oct: the correction is smaller than spread + fee.
Details: strategy/kalshi_corrected_model.xlsx.
