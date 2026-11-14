"""One-time TradeStation login: prints the browser URL, then turns the code into a refresh token.

Usage (TS_CLIENT_ID and TS_CLIENT_SECRET already set in the environment):

    python examples/ts_login.py                 # step 1: prints the login URL
    python examples/ts_login.py <code>          # step 2: paste the ?code=... value from the redirect

Step 2 prints the refresh token ONCE. Store it as TS_REFRESH_TOKEN in the environment
settings; never paste it into chat. Scopes requested: MarketData and ReadAccount only.
"""

import os
import sys

from kalshi_btc_bot.tradestation import auth_url, exchange_code

REDIRECT = os.environ.get("TS_REDIRECT_URI", "http://localhost:3001/callback")


def main() -> None:
    cid, sec = os.environ.get("TS_CLIENT_ID"), os.environ.get("TS_CLIENT_SECRET")
    if not cid or not sec:
        sys.exit("set TS_CLIENT_ID and TS_CLIENT_SECRET first")
    if len(sys.argv) < 2:
        print("Open this in your browser, sign in, approve, then copy the code= value from the address bar:\n")
        print(auth_url(cid, REDIRECT))
        print(f"\nThe page at {REDIRECT} will not load; that is fine, only the code in the URL matters.")
        return
    code = sys.argv[1].split("code=")[-1].split("&")[0]
    tok = exchange_code(cid, sec, code, REDIRECT)
    print("refresh token (store as TS_REFRESH_TOKEN, do not share):\n")
    print(tok["refresh_token"])


if __name__ == "__main__":
    main()
