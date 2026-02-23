import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_roadmap_chart():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # カラー設定
    col_done = '#E0E0E0'      # 完了（グレー）
    col_current = '#FFCC80'   # 現在（薄いオレンジ）
    col_future = '#ED7D31'    # 未来（濃いオレンジ - NLinearカラー）
    
    # 矢印スタイル
    arrow_args = dict(arrowstyle='->', lw=2, color='#555')
    
    # ==========================================
    # Step 1: モデル開発 (完了)
    # ==========================================
    ax.add_patch(patches.FancyBboxPatch((0.5, 3.5), 3, 1.5, boxstyle='round,pad=0.2', 
                                        facecolor=col_done, edgecolor='gray'))
    ax.text(2.0, 4.5, 'Phase 0:\nDevelopment', ha='center', va='center', fontweight='bold', fontsize=12, color='#555')
    ax.text(2.0, 3.9, 'Offline Training\n(Historical Data)', ha='center', va='center', fontsize=9)
    
    # ステータス
    ax.text(2.0, 2.8, '✅ Completed', ha='center', color='green', fontweight='bold')

    # ==========================================
    # Step 2: シャドウモード (現在〜直近)
    # ==========================================
    ax.add_patch(patches.FancyBboxPatch((4.0, 3.5), 3.5, 1.5, boxstyle='round,pad=0.2', 
                                        facecolor=col_current, edgecolor='#E65100', lw=2))
    ax.text(5.75, 4.5, 'Phase 1:\nShadow Mode', ha='center', va='center', fontweight='bold', fontsize=12, color='#E65100')
    ax.text(5.75, 3.9, 'Real-time Prediction\n(No Control)', ha='center', va='center', fontsize=9)
    
    # 重要ポイントの注釈
    ax.annotate('Verify Robustness\nin Summer/Winter', xy=(5.75, 3.5), xytext=(5.75, 2.0),
                arrowprops=dict(arrowstyle='->', color='#E65100'), ha='center', fontsize=10, fontweight='bold', color='#E65100')

    # ==========================================
    # Step 3: 実運用 (未来)
    # ==========================================
    ax.add_patch(patches.FancyBboxPatch((8.0, 3.5), 3.5, 1.5, boxstyle='round,pad=0.2', 
                                        facecolor=col_future, edgecolor='#BF360C'))
    ax.text(9.75, 4.5, 'Phase 2 & 3:\nDeployment', ha='center', va='center', fontweight='bold', fontsize=12, color='white')
    ax.text(9.75, 3.9, 'Closed-Loop Control\n(Automated)', ha='center', va='center', fontsize=9, color='white')

    # メリット
    ax.text(9.75, 2.8, 'Target:\nLife Extension +15y\nCost -30%', ha='center', va='top', fontsize=10, fontweight='bold', color='#BF360C')

    # ==========================================
    # タイムライン矢印
    # ==========================================
    ax.annotate('', xy=(3.8, 4.25), xytext=(3.5, 4.25), arrowprops=arrow_args)
    ax.annotate('', xy=(7.8, 4.25), xytext=(7.5, 4.25), arrowprops=arrow_args)
    
    # 下部の基準線 (ISO規格など)
    ax.plot([0.5, 11.5], [1.0, 1.0], color='gray', linestyle='--', lw=1)
    ax.text(0.5, 1.2, 'Reference Standard:', fontsize=10, fontweight='bold', color='#555')
    ax.text(6.0, 0.7, 'ISO 13374 (Condition Monitoring) Process Guidelines', 
            ha='center', fontsize=10, style='italic', backgroundcolor='#eee')

    # タイトル
    ax.set_title('Implementation Roadmap: From Validation to Value', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('roadmap_slide.png', dpi=300)
    print("Saved roadmap_slide.png")
    plt.show()

if __name__ == "__main__":
    create_roadmap_chart()