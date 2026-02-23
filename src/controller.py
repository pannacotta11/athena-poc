import numpy as np

class ThermalController:
    """
    Hybrid Control Logic for Transformer Oil Temperature
    Controls cooling fans based on Real-time data & AI Prediction.
    """
    def __init__(self, limit_failsafe=42.0, limit_warning=38.0):
        self.limit_failsafe = limit_failsafe # Absolute limit (Hardware)
        self.limit_warning = limit_warning   # AI prediction limit (Software)

    def decide(self, current_temp, future_temps_array):
        """
        決定ロジックを実行する
        Args:
            current_temp (float): 現在のセンサー温度
            future_temps_array (np.array): AIが予測した未来の温度推移
        Returns:
            status_code (int): 0=Normal, 1=AI_PreCool, 2=FailSafe
            message (str): ログ出力用メッセージ
        """
        # 1. Fail-Safe (Priority High)
        if current_temp >= self.limit_failsafe:
            return 2, f"CRITICAL: Temp {current_temp:.2f}°C exceeded Fail-Safe limit!"

        # 2. AI Pre-cooling (Priority Medium)
        max_pred = np.max(future_temps_array)
        
        if max_pred >= self.limit_warning:
            # 何分後に超えるかを計算 (index * 15min)
            # np.argmaxは条件を満たす最初のインデックスを返す
            reach_idx = np.argmax(future_temps_array >= self.limit_warning)
            minutes_left = (reach_idx + 1) * 15
            
            return 1, f"WARNING: AI predicts {max_pred:.2f}°C in {minutes_left} min. Pre-cooling ON."

        # 3. Normal Operation
        return 0, "NORMAL: System within safety range."