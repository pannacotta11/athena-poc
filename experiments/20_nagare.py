import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_operation_flow_chart():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # スタイル設定
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    arrow_props = dict(arrowstyle='->', lw=2, color='#555')
    text_props = dict(ha='center', va='center', fontsize=12, fontweight='bold', color='#333')
    
    # カラー
    col_sensor = '#E3F2FD' # 薄い青
    col_ai = '#FFF3E0'     # 薄いオレンジ (NLinear)
    col_action = '#E8F5E9' # 薄い緑
    
    # ==========================================
    # 1. データ収集 (Sensors)
    # ==========================================
    ax.add_patch(patches.FancyBboxPatch((3, 8.5), 4, 1.2, boxstyle='round,pad=0.2', facecolor=col_sensor, edgecolor='#1E88E5'))
    ax.text(5, 9.1, 'Step 1: Real-time Monitoring', fontsize=11, fontweight='bold', color='#1565C0')
    ax.text(5, 8.7, 'Sensors: Load (HUFL) & Oil Temp (OT)', fontsize=10, ha='center')

    # 矢印
    ax.annotate('', xy=(5, 7.8), xytext=(5, 8.5), arrowprops=arrow_props)

    # ==========================================
    # 2. AI予測 (NLinear)
    # ==========================================
    ax.add_patch(patches.FancyBboxPatch((2, 6.0), 6, 1.8, boxstyle='round,pad=0.2', facecolor=col_ai, edgecolor='#EF6C00'))
    ax.text(5, 7.3, 'Step 2: AI Future Prediction (NLinear)', fontsize=12, fontweight='bold', color='#E65100')
    ax.text(5, 6.7, 'Predict Temp for next 24~96 hours', fontsize=10, ha='center')
    ax.text(5, 6.3, 'Check: "Will it exceed limit?"', fontsize=10, ha='center', style='italic')

    # 矢印 (分岐)
    ax.annotate('', xy=(3, 5.0), xytext=(4, 6.0), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 5.0), xytext=(6, 6.0), arrowprops=arrow_props)
    
    # 分岐テキスト
    ax.text(3, 5.5, 'Normal', ha='center', fontsize=10, color='gray', backgroundcolor='white')
    ax.text(7, 5.5, 'Warning!\n(Overshoot Predicted)', ha='center', fontsize=10, color='red', fontweight='bold', backgroundcolor='white')

    # ==========================================
    # 3. アクション (Normal vs Action)
    # ==========================================
    # Normal Case
    ax.add_patch(patches.FancyBboxPatch((1, 3.5), 4, 1.5, boxstyle='round,pad=0.2', facecolor='#F5F5F5', edgecolor='gray'))
    ax.text(3, 4.4, 'Standard Operation', fontsize=11, fontweight='bold')
    ax.text(3, 3.9, '- Continuous Logging\n- Energy Saving Mode', fontsize=9, ha='center', color='#555')

    # Action Case (ここが重要)
    ax.add_patch(patches.FancyBboxPatch((5.5, 3.0), 4, 2.0, boxstyle='round,pad=0.2', facecolor=col_action, edgecolor='#2E7D32', lw=2))
    ax.text(7.5, 4.5, 'Step 3: Proactive Control', fontsize=12, fontweight='bold', color='#1B5E20')
    ax.text(7.5, 3.8, '1. Pre-Cooling (Start Fans Early)\n2. Load Shedding (If Critical)\n3. Alert Maintenance Team', 
            fontsize=9.5, ha='left', va='center')

    # ==========================================
    # 4. 結果 (Result)
    # ==========================================
    # 矢印合流
    ax.annotate('', xy=(5, 2.0), xytext=(3, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 2.0), xytext=(7.5, 3.0), arrowprops=arrow_props)

    # Result Box
    ax.add_patch(patches.FancyBboxPatch((2.5, 0.5), 5, 1.5, boxstyle='round,pad=0.2', facecolor='white', edgecolor='#333', linestyle='--'))
    ax.text(5, 1.4, 'OUTCOME', fontsize=12, fontweight='bold', color='#333')
    ax.text(5, 0.9, 'Prevent Overheating (Life Extension)\n& Optimize Energy Usage', fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig('operation_flow.png', dpi=300)
    print("Saved operation_flow.png")
    plt.show()

if __name__ == "__main__":
    create_operation_flow_chart()