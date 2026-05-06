
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class vizualization:
    """"""
    def __init__(self,kaiming_dict, xavier_dict,orthogonal_dict,activation_name):
        self.kaiming_dict = kaiming_dict
        self.xavier_dict = xavier_dict
        self.orthogonal_dict = orthogonal_dict
        self.activation_name = activation_name

        records =[]

        all_simulations = {
            'kaiming': self.kaiming_dict,
            'xavier' : self.xavier_dict,
            'orthogonal' : self.orthogonal_dict
        }

        for init_name, res in all_simulations.items():

            for run_idx, run_grads in enumerate(res['gradients']):
                for layer_idx, val in enumerate(run_grads):
                    records.append({
                        'Init':init_name,
                        'Layer': f"Layer {layer_idx + 1}",
                        'Type': 'Grad Norm',
                        'Value':val
                    })

            for run_idx, run_vars in enumerate(res['variances']):
                for layer_idx, val in enumerate(run_vars):
                    records.append({
                        'Init': init_name,
                        'Layer': f"Layer {layer_idx + 1}",
                        'Type': 'Var Ratio',
                        'Value':val
                    })
    
        self.records = pd.DataFrame(records)


    
    def plotBoxPlots(self):

        df = self.records

        sns.set_theme(style='whitegrid')
        fig, axes = plt.subplots(2,1,figsize=(16,8))
        fig.suptitle(f"monte Carlo Analysis" , fontsize = 20, fontweight = 'bold')

        try:
            layer_order = sorted(df['Layer'].unique(), key=lambda x: int(x.split(' ')[-1]))
        except:
            layer_order = df['Layer'].unique()

        palette = {'default': 'gray', 'xavier': 'orange', 'kaiming': 'green', 'orthogonal': 'red'}

        sns.boxplot(data=df[df['Type']=='Grad Norm'], x='Layer', y='Value', hue='Init',
                    order=layer_order, ax=axes[0], palette=palette, showfliers=False)
        axes[0].set_title("Gradient Norms (Backward Pass)")

        sns.boxplot(data=df[df['Type']=='Var Ratio'], x='Layer', y='Value', hue='Init',
                    order=layer_order, ax=axes[1], palette=palette, showfliers=False)
        axes[1].set_title("Variance Ratio (Forward Pass)")
        axes[1].axhline(1, color='red', linestyle='--')

        plt.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig(f"boxplots_{self.activation_name}.png", dpi=150, bbox_inches='tight')
        plt.show()






    # def plotDiagnostics(self):
        
    #     num_layers = len(self.kaiming_dict['gradients'][0])
    
    #     print(f"Detektovano {num_layers} slojeva za vizuelizaciju.")

    # # ---------------------------------------------------------
    # #            Preparation
    # # ---------------------------------------------------------
    #     fig_width = max(12, num_layers * 3) 
    #     fig = plt.figure(figsize=(fig_width, 18))
    
    # # GridSpec: Gornji redovi za histograme, donji (4) za sumarni grafik
    #     gs = fig.add_gridspec(4, num_layers, height_ratios=[1, 1, 1, 1.5])

    #     fig.suptitle(f"Single-Run Dijagnostika: {self.activation_name.upper()}", fontsize=24, fontweight='bold', y=0.98)

    #     cache_grads = {}
    #     cache_vars = {}

    #     print(self.records.head(200).to_string())

    # ---------------------------------------------------------
    # KORAK 3: ANALIZA PO INICIJALIZACIJAMA
    # ---------------------------------------------------------
        #for init_type, res in self.all_simulation.items():
            #print(f"---- Analiza inicijalizacije: {init_type} ---")

            #torch.manual_seed(42) # Fiksiran seed da bi poreÄ‘enje bilo fer

#            itype = init_type

            # cache_grads[init_type] = 
            # cache_vars[init_type] = vars_r

    #     # CRTAJ HISTOGRAME (Za svaki sloj u trenutnom redu 'row')
    #     for col, layer_name in enumerate(layer_names):
    #         if col >= num_layers: break 
            
    #         ax = fig.add_subplot(gs[row, col])
    #         data = acts[layer_name]
            
    #         mean, std = np.mean(data), np.std(data)
    #         colors = {'default': 'gray', 'xavier': 'orange', 'kaiming': 'green', 'orthogonal': 'red'}
            
    #         ax.hist(data, bins=50, color=colors[init_type], alpha=0.6)
    #         ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    #         ax.set_xlim(-5, 5) # Fokus na centar
    #         ax.set_yticks([])
            
    #         # Naslovi
    #         title_txt = ""
    #         if row == 0: title_txt += f"=== {layer_name} ===\n"
    #         title_txt += f"Mean:{mean:.2f}, Std:{std:.2f}"
    #         ax.set_title(title_txt, fontsize=10)

    #         # Oznaka inicijalizacije levo
    #         if col == 0:
    #             ax.set_ylabel(f"{init_type.upper()}", fontsize=12, fontweight='bold', rotation=90, labelpad=10)

    # # ---------------------------------------------------------
    # # KORAK 4: GRADIJENTI I VARIJANSE (DONJI RED)
    # # ---------------------------------------------------------
    # mid_point = max(1, num_layers // 2)
    # # Spajamo kolone za donje grafike
    # ax_grad = fig.add_subplot(gs[4, :mid_point])
    # ax_var = fig.add_subplot(gs[4, mid_point:])
    
    # # Pretpostavljamo da imamo gradijent za svaki sloj
    # # Ponekad output layer nema activation hook, pa pazimo na duÅ¾ine
    
    # for init_type in init_types:
    #     norms = cache_grads[init_type]
    #     ratios = cache_vars[init_type]
        
    #     # X osa
    #     x_axis_grad = range(1, len(norms) + 1)
    #     x_axis_var = range(1, len(ratios) + 1)
        
    #     ax_grad.plot(x_axis_grad, norms, marker='o', linewidth=2, label=init_type.upper())
    #     ax_var.plot(x_axis_var, ratios, marker='s', linewidth=2, label=init_type.upper())

    # ax_grad.set_title("Gradient norms (Backward Pass)", fontsize=14)
    # ax_grad.set_xlabel("Layer Index")
    # ax_grad.grid(True, alpha=0.5)
    # ax_grad.legend()

    # ax_var.set_title("Variance ratio (Forward Pass)", fontsize=14)
    # ax_var.set_xlabel("Layer Index")
    # ax_var.axhline(1, color='red', linestyle='--', label="Ideal")
    # ax_var.grid(True, alpha=0.5)
    # ax_var.legend()

    # plt.tight_layout()
    # fig.subplots_adjust(top=0.92) # Mesto za glavni naslov
    # plt.show()