import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BitcoinHighSMA200Strategy:
    def __init__(self, symbol='BTC-USD', initial_capital=100000):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.results = None
        
    def fetch_data(self, period='2y'):
        """獲取比特幣歷史數據"""
        try:
            self.data = yf.download(self.symbol, period=period, interval='1d')
            # 如果是多層索引，展平列名
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = [col[0] for col in self.data.columns]
            print(f"成功獲取 {self.symbol} 數據，共 {len(self.data)} 天")
            return True
        except Exception as e:
            print(f"獲取數據失敗: {e}")
            return False
    
    def calculate_indicators(self):
        """計算技術指標"""
        df = self.data.copy()
        
        # SMA200
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # 歷史最高點
        df['Historical_High'] = df['Close'].expanding().max()
        
        # 檢測新高點
        df['New_High'] = df['Close'] == df['Historical_High']
        df['High_Change'] = df['Historical_High'] != df['Historical_High'].shift(1)
        
        # 局部高點檢測（使用20日窗口）
        df['Local_High'] = df['High'].rolling(window=21, center=True).max() == df['High']
        
        # 修正前高邏輯：將局部高點也當作前高處理
        df['Previous_High'] = 0.0
        current_prev_high = 0.0
        
        for i in range(len(df)):
            # 檢查是否為歷史最高點變化或局部高點
            is_historical_high_change = df.iloc[i]['High_Change'] and i > 0
            is_local_high = bool(df.iloc[i]['Local_High']) if df.iloc[i]['Local_High'] == df.iloc[i]['Local_High'] else False
            
            # 如果是歷史最高點變化，前高更新為前一個歷史最高點
            if is_historical_high_change:
                current_prev_high = df.iloc[i-1]['Historical_High']
            # 如果是局部高點且不是當前歷史最高點，將其設為前高
            elif is_local_high and df.iloc[i]['Close'] != df.iloc[i]['Historical_High']:
                # 只有當局部高點大於當前前高時才更新
                if df.iloc[i]['High'] > current_prev_high:
                    current_prev_high = df.iloc[i]['High']
            
            df.iloc[i, df.columns.get_loc('Previous_High')] = current_prev_high
        
        # 從歷史最高點的回調百分比
        df['Drawdown_From_High'] = (df['Close'] - df['Historical_High']) / df['Historical_High'] * 100
        
        # 從前高的回調百分比
        df['Drawdown_From_Previous_High'] = np.where(
            df['Previous_High'] > 0,
            (df['Close'] - df['Previous_High']) / df['Previous_High'] * 100,
            0
        )
        
        # 檢查歷史最高點是否在SMA200以上
        df['High_Above_SMA200'] = df['Historical_High'] > df['SMA_200']
        
        # 檢查前高是否在SMA200以上
        df['Previous_High_Above_SMA200'] = df['Previous_High'] > df['SMA_200']
        
        # 成交量指標
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # 新高確認（價格突破前期高點）
        df['New_High'] = df['Close'] == df['Historical_High']
        
        # 動態資金計算（複利）
        df['Available_Capital'] = 0.0
        
        self.data = df
        
    def generate_signals(self):
        """生成交易信號"""
        df = self.data.copy()
        
        # 初始化信號和倉位
        df['Signal'] = 0
        df['Position_30'] = 0.0  # 30%倉位
        df['Position_60'] = 0.0  # 60%倉位
        df['Total_Position'] = 0.0  # 總倉位
        df['Entry_Price_30'] = 0.0
        df['Entry_Price_60'] = 0.0
        df['Historical_High_Entry'] = 0.0  # 進場時的歷史最高點
        df['Previous_High_Entry'] = 0.0    # 進場時的前高
        df['Current_Capital'] = self.initial_capital  # 當前可用資金（複利）
        df['Position_Value'] = 0.0  # 持倉價值
        df['Cash_Balance'] = self.initial_capital  # 現金餘額
        
        # 記錄交易決策
        df['Trade_Reason'] = ''
        
        for i in range(200, len(df)):  # 從第200天開始，確保SMA200計算完整
            current_price = df.iloc[i]['Close']
            drawdown = df.iloc[i]['Drawdown_From_High']
            historical_high = df.iloc[i]['Historical_High']
            new_high = df.iloc[i]['New_High']
            high_above_sma200 = df.iloc[i]['High_Above_SMA200']
            
            # 獲取前一天的狀態
            prev_pos_30 = df.iloc[i-1]['Position_30']
            prev_pos_60 = df.iloc[i-1]['Position_60']
            prev_total_pos = df.iloc[i-1]['Total_Position']
            prev_historical_high_entry = df.iloc[i-1]['Historical_High_Entry']
            prev_current_capital = df.iloc[i-1]['Current_Capital']
            prev_cash_balance = df.iloc[i-1]['Cash_Balance']
            
            # 更新持倉價值
            total_shares = prev_pos_30 + prev_pos_60
            if total_shares > 0:
                position_value = total_shares * current_price
            else:
                position_value = 0
            
            current_total_value = prev_cash_balance + position_value
            
            # 默認保持前一天的狀態
            df.iloc[i, df.columns.get_loc('Position_30')] = prev_pos_30
            df.iloc[i, df.columns.get_loc('Position_60')] = prev_pos_60
            df.iloc[i, df.columns.get_loc('Total_Position')] = prev_total_pos
            df.iloc[i, df.columns.get_loc('Entry_Price_30')] = df.iloc[i-1]['Entry_Price_30']
            df.iloc[i, df.columns.get_loc('Entry_Price_60')] = df.iloc[i-1]['Entry_Price_60']
            df.iloc[i, df.columns.get_loc('Historical_High_Entry')] = prev_historical_high_entry
            df.iloc[i, df.columns.get_loc('Previous_High_Entry')] = df.iloc[i-1]['Previous_High_Entry']
            df.iloc[i, df.columns.get_loc('Current_Capital')] = current_total_value
            df.iloc[i, df.columns.get_loc('Position_Value')] = position_value
            df.iloc[i, df.columns.get_loc('Cash_Balance')] = prev_cash_balance
            
            # 檢查止損條件
            should_stop_loss = False
            stop_loss_reason = ""
            
            # 檢查30%倉位止損（從進場價跌20%）
            if prev_pos_30 > 0 and df.iloc[i-1]['Entry_Price_30'] > 0:
                stop_loss_30 = df.iloc[i-1]['Entry_Price_30'] * 0.8  # 跌20%
                if current_price <= stop_loss_30:
                    should_stop_loss = True
                    stop_loss_reason = f"30% position stop loss at {current_price:.2f} (from {df.iloc[i-1]['Entry_Price_30']:.2f})"
            
            # 檢查60%倉位止損（從進場價跌10%）
            if prev_pos_60 > 0 and df.iloc[i-1]['Entry_Price_60'] > 0:
                stop_loss_60 = df.iloc[i-1]['Entry_Price_60'] * 0.9  # 跌10%
                if current_price <= stop_loss_60:
                    should_stop_loss = True
                    stop_loss_reason = f"60% position stop loss at {current_price:.2f} (from {df.iloc[i-1]['Entry_Price_60']:.2f})"
            
            if should_stop_loss:
                # 觸發止損，清倉
                df.iloc[i, df.columns.get_loc('Signal')] = -1
                df.iloc[i, df.columns.get_loc('Position_30')] = 0.0
                df.iloc[i, df.columns.get_loc('Position_60')] = 0.0
                df.iloc[i, df.columns.get_loc('Total_Position')] = 0.0
                df.iloc[i, df.columns.get_loc('Entry_Price_30')] = 0.0
                df.iloc[i, df.columns.get_loc('Entry_Price_60')] = 0.0
                df.iloc[i, df.columns.get_loc('Historical_High_Entry')] = 0.0
                df.iloc[i, df.columns.get_loc('Previous_High_Entry')] = 0.0
                df.iloc[i, df.columns.get_loc('Cash_Balance')] = current_total_value
                df.iloc[i, df.columns.get_loc('Position_Value')] = 0.0
                df.iloc[i, df.columns.get_loc('Trade_Reason')] = stop_loss_reason
                continue
            
            # 檢查止盈條件
            should_take_profit = False
            take_profit_reason = ""
            
            if prev_total_pos > 0:
                # 條件1：創新高時止盈
                if new_high:
                    should_take_profit = True
                    take_profit_reason = f'Take Profit: New High at {current_price:.2f}'
                
                # 條件2：從前高反彈超過10%時止盈（積極止盈）
                elif previous_high > 0 and drawdown_from_prev_high >= -10:
                    should_take_profit = True
                    take_profit_reason = f'Take Profit: Rebound to -10% from previous high {previous_high:.2f}'
                
                # 條件3：持倉超過30天且有盈利時止盈
                elif prev_previous_high_entry > 0:
                    # 找到最早的進場日期
                    entry_dates = []
                    if df.iloc[i-1]['Entry_Price_30'] > 0:
                        entry_dates.append(df.iloc[i-1]['Entry_Price_30'])
                    if df.iloc[i-1]['Entry_Price_60'] > 0:
                        entry_dates.append(df.iloc[i-1]['Entry_Price_60'])
                    
                    # 如果持倉30天且當前價格高於前高的90%，止盈
                    if i >= 230 and previous_high > 0 and current_price > previous_high * 0.9:  # 30天 + 200天初始期
                        days_held = 30  # 簡化計算
                        if days_held >= 30:
                            should_take_profit = True
                            take_profit_reason = f'Take Profit: Held 30+ days with good position at {current_price:.2f}'
            
            if should_take_profit:
                # 執行止盈
                df.iloc[i, df.columns.get_loc('Signal')] = -1
                df.iloc[i, df.columns.get_loc('Position_30')] = 0.0
                df.iloc[i, df.columns.get_loc('Position_60')] = 0.0
                df.iloc[i, df.columns.get_loc('Total_Position')] = 0.0
                df.iloc[i, df.columns.get_loc('Entry_Price_30')] = 0.0
                df.iloc[i, df.columns.get_loc('Entry_Price_60')] = 0.0
                df.iloc[i, df.columns.get_loc('Historical_High_Entry')] = 0.0
                df.iloc[i, df.columns.get_loc('Previous_High_Entry')] = 0.0
                df.iloc[i, df.columns.get_loc('Cash_Balance')] = current_total_value
                df.iloc[i, df.columns.get_loc('Position_Value')] = 0.0
                df.iloc[i, df.columns.get_loc('Trade_Reason')] = take_profit_reason
                continue
            
            # 進場邏輯（只有前高在SMA200以上時才進場）
            previous_high = df.iloc[i]['Previous_High']
            prev_high_above_sma200 = df.iloc[i]['Previous_High_Above_SMA200']
            drawdown_from_prev_high = df.iloc[i]['Drawdown_From_Previous_High']
            prev_previous_high_entry = df.iloc[i-1]['Previous_High_Entry'] if i > 0 else 0
            
            if prev_high_above_sma200 and previous_high > 0:
                # 檢查是否滿足進場條件（基於前高回調）
                # 修改邏輯：允許在新的前高下進場，即使已有持倉
                entry_20_condition = (drawdown_from_prev_high <= -20 and drawdown_from_prev_high > -30)
                entry_30_condition = (drawdown_from_prev_high <= -30)
                
                # 第一次進場條件：前高回調20%，無持倉時投入30%資金
                if entry_20_condition and prev_total_pos == 0:
                    investment_amount = current_total_value * 0.9
                    shares_to_buy = investment_amount / current_price
                    
                    df.iloc[i, df.columns.get_loc('Signal')] = 1
                    df.iloc[i, df.columns.get_loc('Position_30')] = shares_to_buy
                    df.iloc[i, df.columns.get_loc('Total_Position')] = shares_to_buy
                    df.iloc[i, df.columns.get_loc('Entry_Price_30')] = current_price
                    df.iloc[i, df.columns.get_loc('Historical_High_Entry')] = historical_high
                    df.iloc[i, df.columns.get_loc('Previous_High_Entry')] = previous_high
                    df.iloc[i, df.columns.get_loc('Cash_Balance')] = current_total_value - investment_amount
                    df.iloc[i, df.columns.get_loc('Position_Value')] = investment_amount
                    
                    df.iloc[i, df.columns.get_loc('Trade_Reason')] = f'Entry 30% at 20% drawdown from previous high {previous_high:.2f}: Current {current_price:.2f}, Investment: ${investment_amount:.2f}'
                

        self.signals = df
        
    def calculate_returns(self):
        """計算策略回報"""
        df = self.signals.copy()
        
        # 計算每日收益率
        df['Daily_Return'] = df['Close'].pct_change()
        
        # 計算策略收益率（基於總資產變化）
        df['Portfolio_Value'] = df['Current_Capital']
        df['Strategy_Daily_Return'] = df['Portfolio_Value'].pct_change()
        
        # 計算累積收益
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital
        df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod()
        
        # 計算回撤
        df['Peak'] = df['Portfolio_Value'].expanding().max()
        df['Drawdown'] = (df['Portfolio_Value'] - df['Peak']) / df['Peak'] * 100
        
        self.results = df
        
    def analyze_performance(self):
        """分析策略表現"""
        df = self.results.copy()
        
        # 基本統計
        entry_trades = len(df[df['Signal'] == 1])
        exit_trades = len(df[df['Signal'] == -1])
        total_return = (df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # 計算年化報酬率
        trading_days = len(df.dropna())
        years = trading_days / 252
        cagr = (df['Portfolio_Value'].iloc[-1] / self.initial_capital) ** (1/years) - 1
        
        # 最大回撤
        max_drawdown = df['Drawdown'].min()
        
        # 夏普比率
        strategy_returns = df['Strategy_Daily_Return'].dropna()
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 交易分析
        trade_analysis = self.analyze_trades()
        
        # 波動率
        volatility = strategy_returns.std() * np.sqrt(252) if len(strategy_returns) > 0 else 0
        
        # 市場對比
        market_return = (df['Cumulative_Market_Return'].iloc[-1] - 1) * 100
        market_cagr = (df['Cumulative_Market_Return'].iloc[-1]) ** (1/years) - 1
        
        # 在倉時間比例
        time_in_market = len(df[df['Total_Position'] > 0]) / len(df) * 100
        
        # 資金利用率
        max_investment = df['Position_Value'].max()
        capital_efficiency = max_investment / self.initial_capital * 100
        
        performance_metrics = {
            'Entry Trades': entry_trades,
            'Exit Trades': exit_trades,
            'Total Return (%)': round(total_return, 2),
            'CAGR (%)': round(cagr * 100, 2),
            'Market CAGR (%)': round(market_cagr * 100, 2),
            'Alpha (%)': round((cagr - market_cagr) * 100, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Volatility (%)': round(volatility * 100, 2),
            'Time in Market (%)': round(time_in_market, 2),
            'Capital Efficiency (%)': round(capital_efficiency, 2),
            'Final Portfolio Value': round(df['Portfolio_Value'].iloc[-1], 2),
            **trade_analysis
        }
        
        return performance_metrics
    
    def analyze_trades(self):
        """分析具體交易情況"""
        df = self.results.copy()
        
        # 統計交易原因
        trade_reasons = df[df['Trade_Reason'] != '']['Trade_Reason'].value_counts()
        
        # 統計進場次數
        total_entries = len(df[df['Signal'] == 1])
        
        # 找出所有完整的交易週期
        trades = []
        current_trade = None
        
        for i, row in df.iterrows():
            if row['Signal'] == 1:  # 進場
                if current_trade is None:
                    current_trade = {
                        'entry_date': i,
                        'entry_price': row['Close'],
                        'entry_reason': row['Trade_Reason'],
                        'entry_value': row['Position_Value']
                    }
                else:
                    # 加倉
                    current_trade['entry_value'] = row['Position_Value']
            elif row['Signal'] == -1 and current_trade is not None:  # 出場
                current_trade.update({
                    'exit_date': i,
                    'exit_price': row['Close'],
                    'exit_reason': row['Trade_Reason'],
                    'exit_value': row['Current_Capital'],
                    'duration': (i - current_trade['entry_date']).days,
                    'return': (row['Current_Capital'] - current_trade['entry_value']) / current_trade['entry_value'] if current_trade['entry_value'] > 0 else 0
                })
                trades.append(current_trade)
                current_trade = None
        
        # 分析交易結果
        if trades:
            winning_trades = [t for t in trades if t['return'] > 0]
            losing_trades = [t for t in trades if t['return'] < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['return'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            avg_duration = np.mean([t['duration'] for t in trades])
        else:
            win_rate = 0
            profit_factor = 0
            avg_duration = 0
        
        return {
            'Win Rate (%)': round(win_rate * 100, 2),
            'Profit Factor': round(profit_factor, 2),
            'Average Trade Duration (days)': round(avg_duration, 2),
            'Total Completed Trades': len(trades)
        }
    
    def plot_results(self):
        """繪製回測結果圖表"""
        fig, axes = plt.subplots(5, 1, figsize=(15, 20))
        
        # 1. 價格走勢與前高、歷史最高點及SMA200
        axes[0].plot(self.results.index, self.results['Close'], label='BTC價格', linewidth=1.5, color='black')
        axes[0].plot(self.results.index, self.results['Historical_High'], label='歷史最高點', alpha=0.7, color='red', linestyle='-')
        
        # 繪製前高（包含局部高點）
        prev_high_data = self.results[self.results['Previous_High'] > 0]
        if not prev_high_data.empty:
            axes[0].plot(prev_high_data.index, prev_high_data['Previous_High'], 
                        label='前高(含局部高點)', linewidth=2, color='orange', alpha=0.8, linestyle='--')
        
        axes[0].plot(self.results.index, self.results['SMA_200'], label='SMA 200', alpha=0.7, color='blue')
        
        # 標示交易點
        entry_signals = self.results[self.results['Signal'] == 1]
        exit_signals = self.results[self.results['Signal'] == -1]
        
        axes[0].scatter(entry_signals.index, entry_signals['Close'], color='green', marker='^', s=100, label='Entry', zorder=5)
        axes[0].scatter(exit_signals.index, exit_signals['Close'], color='red', marker='v', s=100, label='Exit', zorder=5)
        
        axes[0].set_title('Bitcoin前高(含局部高點)回調策略 - 複利增長')
        axes[0].set_ylabel('Price (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 從前高的回調分析
        axes[1].plot(self.results.index, self.results['Drawdown_From_Previous_High'], color='purple', linewidth=1)
        axes[1].axhline(y=-20, color='orange', linestyle='--', alpha=0.7, label='20%回調 (30%資金)')
        axes[1].axhline(y=-30, color='red', linestyle='--', alpha=0.7, label='30%回調 (再30%資金)')
        axes[1].fill_between(self.results.index, self.results['Drawdown_From_Previous_High'], 0, alpha=0.3, color='purple')
        axes[1].set_title('從前高的回調分析')
        axes[1].set_ylabel('回調 (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 策略表現對比
        axes[2].plot(self.results.index, self.results['Cumulative_Return'], label='High SMA200 Strategy (Compound)', linewidth=2, color='blue')
        axes[2].plot(self.results.index, self.results['Cumulative_Market_Return'], label='Buy & Hold', linewidth=2, color='gray')
        axes[2].set_title('Strategy Performance vs Buy & Hold (Compound Growth)')
        axes[2].set_ylabel('Cumulative Return')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. 資金分配圖
        axes[3].fill_between(self.results.index, 0, self.results['Position_Value'], alpha=0.6, color='green', label='Position Value')
        axes[3].fill_between(self.results.index, self.results['Position_Value'], self.results['Current_Capital'], alpha=0.4, color='blue', label='Cash Balance')
        axes[3].plot(self.results.index, self.results['Current_Capital'], color='red', linewidth=2, label='Total Capital')
        axes[3].set_title('Capital Allocation Over Time (Compound Growth)')
        axes[3].set_ylabel('Value (USD)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 5. 策略回撤
        axes[4].fill_between(self.results.index, self.results['Drawdown'], 0, alpha=0.3, color='red')
        axes[4].plot(self.results.index, self.results['Drawdown'], color='darkred', linewidth=1)
        axes[4].set_title('Strategy Drawdown')
        axes[4].set_ylabel('Drawdown (%)')
        axes[4].set_xlabel('Date')
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('btc_high_sma200_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def print_trade_log(self):
        """打印詳細交易日誌"""
        df = self.results.copy()
        trades = df[df['Trade_Reason'] != '']
        
        print("\n=== 📋 詳細交易日誌 ===")
        print(f"{'日期':<12} {'價格':<10} {'動作':<8} {'資金':<12} {'總資產':<12} {'原因':<60}")
        print("-" * 120)
        
        for i, row in trades.iterrows():
            date = i.strftime('%Y-%m-%d')
            price = f"${row['Close']:,.2f}"
            action = "買入" if row['Signal'] == 1 else "賣出"
            position_value = f"${row['Position_Value']:,.0f}"
            total_capital = f"${row['Current_Capital']:,.0f}"
            reason = row['Trade_Reason']
            print(f"{date:<12} {price:<10} {action:<8} {position_value:<12} {total_capital:<12} {reason:<60}")
    
    def run_backtest(self):
        """執行完整回測"""
        print("開始前高(含局部高點)回調複利策略回測...")
        
        # 獲取數據
        if not self.fetch_data():
            return None
            
        # 計算指標
        print("計算技術指標...")
        self.calculate_indicators()
        
        # 生成信號
        print("生成交易信號...")
        self.generate_signals()
        
        # 計算回報
        print("計算策略回報...")
        self.calculate_returns()
        
        # 分析表現
        print("分析策略表現...")
        performance = self.analyze_performance()
        
        # 繪製圖表
        print("繪製結果圖表...")
        self.plot_results()
        
        # 打印交易日誌
        self.print_trade_log()
        
        return performance

# 執行回測
if __name__ == "__main__":
    strategy = BitcoinHighSMA200Strategy()
    results = strategy.run_backtest()
    
    if results:
        print("\n=== 🚀 前高(含局部高點)回調複利策略回測結果 ===")
        for key, value in results.items():
            print(f"{key}: {value}")
