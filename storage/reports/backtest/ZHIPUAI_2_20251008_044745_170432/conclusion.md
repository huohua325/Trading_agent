# Backtest Conclusion Report

**Backtest Period**: 2025-03-03 ~ 2025-06-30

## Metric Values

- **cum_return**: 0.001991
- **max_drawdown**: -0.139132
- **volatility_daily**: 0.015397
- **sortino**: 0.011815
- **trades_count**: 318
- **trades_notional**: 342862.457543
- **volatility**: 0.244414
- **sharpe**: 0.145613
- **sortino_annual**: 0.187565
- **excess_return_total**: -0.007178
- **tracking_error_daily**: 0.017076
- **information_ratio_daily**: -0.105094
- **beta**: -0.366605
- **corr**: -0.910426
- **up_capture**: -0.719026
- **down_capture**: -0.179210
- **hit_ratio_active**: 0.250000
- **sortino_excess**: -3.862384
- **rolling_ir_63**: nan
- **rolling_te_63**: nan
- **rolling_ir_126**: nan
- **rolling_te_126**: nan
- **excess_return_annual**: -0.452244
- **tracking_error**: 0.271077
- **information_ratio**: -1.668323
- **alpha_simple**: -0.452244
- **rolling_ir_252**: nan
- **rolling_te_252**: nan
- **n**: 4
- **freq**: day

## Metrics Concept Explanation

- **cum_return**: Cumulative return, the rise/fall of final NAV relative to initial NAV
- **max_drawdown**: Maximum drawdown, the maximum decline of NAV relative to historical peak
- **volatility_daily**: Daily volatility, standard deviation of daily returns
- **sortino**: Sortino ratio, average return divided by downside volatility
- **trades_count**: Number of trades, total trading count during backtest period
- **trades_notional**: Total trading amount, sum of all trade notional amounts
- **volatility**: Annualized volatility, daily volatility annualized (×√252)
- **sharpe**: Sharpe ratio, annualized return divided by annualized volatility
- **sortino_annual**: Annualized Sortino ratio, annualized return divided by annualized downside volatility
- **excess_return_total**: Total excess return, cumulative excess return of strategy relative to benchmark
- **tracking_error_daily**: Daily tracking error, daily standard deviation of excess returns
- **information_ratio_daily**: Daily information ratio, average excess return divided by daily tracking error
- **beta**: Beta coefficient, systematic risk coefficient of strategy relative to benchmark
- **corr**: Correlation coefficient, degree of linear correlation between strategy and benchmark returns
- **up_capture**: Upside capture, average capture ratio of strategy when benchmark rises
- **down_capture**: Downside capture, average capture ratio of strategy when benchmark falls
- **hit_ratio_active**: Active hit ratio, proportion of trading days with positive excess returns
- **sortino_excess**: Sortino ratio-excess, excess return divided by downside volatility
- **rolling_ir_63**: Rolling information ratio (63 days), information ratio of short-term rolling window
- **rolling_te_63**: Rolling tracking error (63 days), tracking error of short-term rolling window
- **rolling_ir_126**: Rolling information ratio (126 days), information ratio of medium-term rolling window
- **rolling_te_126**: Rolling tracking error (126 days), tracking error of medium-term rolling window
- **excess_return_annual**: Annualized excess return, annualized excess return rate of strategy relative to benchmark
- **tracking_error**: Annualized tracking error, annualized standard deviation of excess returns
- **information_ratio**: Annualized information ratio, annualized excess return divided by annualized tracking error
- **alpha_simple**: Simple alpha, equivalent to annualized excess return
- **rolling_ir_252**: Rolling information ratio (252 days), information ratio of long-term rolling window
- **rolling_te_252**: Rolling tracking error (252 days), tracking error of long-term rolling window
- **n**: Sample size, number of effective data points
- **freq**: Frequency, data frequency identifier