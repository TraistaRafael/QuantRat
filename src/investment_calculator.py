monthly_in = 4000 + 9000
monthly_grow = 0.1
monthly_cost = 200  # strategy + vps

months = 5


equity = 6000

for i in range(months):
    equity += equity * monthly_grow
    print ("Revenue: {}".format(equity * monthly_grow))
    equity += monthly_in
    equity -= monthly_cost
    print ("Equity: {}".format(equity))




print ("EUR: {}".format(equity / 4.8))