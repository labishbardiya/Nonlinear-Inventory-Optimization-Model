import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class InventoryOptimizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Furniture Shop Inventory Optimization")
        self.root.geometry("1200x800")
        
        # Initialize item_frames dictionary
        self.item_frames = {}
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for inputs
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right panel for results
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Initialize data
        self.initialize_data()
        
        # Create input widgets
        self.create_input_widgets()
        
        # Create result widgets
        self.create_result_widgets()
        
        # Create visualization widgets
        self.create_visualization_widgets()
        
        # Run initial optimization
        self.optimize()
    
    def initialize_data(self):
        # Default values (to be updated dynamically if needed)
        self.items = ["Bed", "Sofa", "Table", "Office Chair", "Almirah", "Stool", "Shoe Rack", 
                     "Dressing Table", "Dining Table", "Study Table", "Pillow", "Mattress", 
                     "Bedsheet", "Wooden Mandir", "Plastic Chair"]
        
        self.D = np.array([72, 84, 120, 120, 96, 120, 84, 60, 24, 84, 240, 300, 60, 36, 1200])  # Annual demand
        self.O = np.array([500, 500, 400, 300, 400, 200, 200, 300, 400, 300, 100, 500, 100, 200, 150])  # Ordering costs
        self.H = np.array([300, 400, 250, 200, 250, 100, 100, 200, 250, 200, 50, 400, 50, 150, 75])  # Holding costs
        self.V = np.array([50, 70, 60, 40, 30, 20, 30, 40, 50, 40, 10, 20, 10, 30, 10])  # Storage volume per unit
        self.L = np.array([30, 30, 30, 30, 45, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30])  # Lead times
        self.MOQ = np.array([2, 2, 2, 3, 1, 3, 3, 2, 2, 2, 5, 2, 5, 2, 10])  # Minimum Order Quantities
        
        self.storage_capacity = 2000  # Initial storage capacity
        self.penalty_factor = 50  # Initial penalty factor
    
    def create_input_widgets(self):
        # Storage capacity slider
        ttk.Label(self.left_panel, text="Storage Capacity (cubic feet):").pack(pady=5)
        self.storage_slider = ttk.Scale(self.left_panel, from_=1000, to=5000, orient=tk.HORIZONTAL)
        self.storage_slider.set(self.storage_capacity)
        self.storage_slider.pack(fill=tk.X, padx=5, pady=5)
        self.storage_slider.bind("<ButtonRelease-1>", lambda e: self.update_storage_capacity(self.storage_slider.get()))
        
        # Penalty factor slider
        ttk.Label(self.left_panel, text="Understocking Penalty Factor:").pack(pady=5)
        self.penalty_slider = ttk.Scale(self.left_panel, from_=10, to=200, orient=tk.HORIZONTAL)
        self.penalty_slider.set(self.penalty_factor)
        self.penalty_slider.pack(fill=tk.X, padx=5, pady=5)
        self.penalty_slider.bind("<ButtonRelease-1>", lambda e: self.update_penalty_factor(self.penalty_slider.get()))
        
        # Create notebook for item-specific inputs
        self.notebook = ttk.Notebook(self.left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frames for each item
        for item in self.items:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=item)
            self.item_frames[item] = frame
            
            # Demand input
            ttk.Label(frame, text="Annual Demand:").pack(pady=2)
            demand_var = tk.StringVar(value=str(self.D[self.items.index(item)]))
            ttk.Entry(frame, textvariable=demand_var).pack(pady=2)
            
            # Ordering cost input
            ttk.Label(frame, text="Ordering Cost (Rs.):").pack(pady=2)
            order_var = tk.StringVar(value=str(self.O[self.items.index(item)]))
            ttk.Entry(frame, textvariable=order_var).pack(pady=2)
            
            # Holding cost input
            ttk.Label(frame, text="Holding Cost (Rs.):").pack(pady=2)
            hold_var = tk.StringVar(value=str(self.H[self.items.index(item)]))
            ttk.Entry(frame, textvariable=hold_var).pack(pady=2)
            
            # Store variables
            self.item_frames[item].vars = {
                'demand': demand_var,
                'order': order_var,
                'hold': hold_var
            }
        
        # Optimize button
        ttk.Button(self.left_panel, text="Optimize", command=self.optimize).pack(pady=10)
    
    def create_result_widgets(self):
        # Create result text widget
        self.result_text = tk.Text(self.right_panel, height=15, width=50)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create cost breakdown frame
        self.cost_frame = ttk.Frame(self.right_panel)
        self.cost_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create labels for cost breakdown
        self.cost_labels = {
            'order': ttk.Label(self.cost_frame, text="Ordering Cost: Rs. 0.00"),
            'hold': ttk.Label(self.cost_frame, text="Holding Cost: Rs. 0.00"),
            'penalty': ttk.Label(self.cost_frame, text="Penalty Cost: Rs. 0.00"),
            'total': ttk.Label(self.cost_frame, text="Total Cost: Rs. 0.00")
        }
        
        for label in self.cost_labels.values():
            label.pack(pady=2)
    
    def create_visualization_widgets(self):
        # Create figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def update_storage_capacity(self, value):
        try:
            self.storage_capacity = float(value)
            self.optimize()
        except Exception as e:
            print(f"Error updating storage capacity: {e}")
    
    def update_penalty_factor(self, value):
        try:
            self.penalty_factor = float(value)
            self.optimize()
        except Exception as e:
            print(f"Error updating penalty factor: {e}")
    
    def get_input_values(self):
        try:
            # Update arrays with current input values
            for i, item in enumerate(self.items):
                if item in self.item_frames and hasattr(self.item_frames[item], 'vars'):
                    self.D[i] = float(self.item_frames[item].vars['demand'].get())
                    self.O[i] = float(self.item_frames[item].vars['order'].get())
                    self.H[i] = float(self.item_frames[item].vars['hold'].get())
        except Exception as e:
            print(f"Error getting input values: {e}")
    
    def total_cost(self, Q):
        if np.any(Q <= 0):
            return np.inf
        
        Q = np.maximum(Q, self.MOQ)
        daily_demand = self.D / 365
        ROP = daily_demand * self.L * 1.5
        safety_stock = (self.D / 12) * 0.3
        avg_inventory = Q/2 + safety_stock
        
        order_cost = (self.D / Q) * self.O
        hold_cost = avg_inventory * self.H
        understock_penalty = np.sum(np.maximum(0, self.D - Q)) * self.penalty_factor
        
        return np.sum(order_cost + hold_cost) + understock_penalty
    
    def storage_constraint(self, Q):
        return self.storage_capacity - np.sum(Q * self.V)
    
    def demand_constraint(self, Q):
        return Q - self.D
    
    def optimize(self):
        try:
            # Get current input values
            self.get_input_values()
            
            # Define constraints and bounds
            constraints = [
                {'type': 'ineq', 'fun': self.storage_constraint},
                {'type': 'ineq', 'fun': self.demand_constraint}
            ]
            bounds = [(m, None) for m in self.MOQ]
            
            # Initial guess
            Q0 = self.MOQ * 2
            
            # Run optimization
            result = minimize(self.total_cost, Q0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            # Update results
            self.update_results(result)
            
            # Update visualization
            self.update_visualization(result)
        except Exception as e:
            print(f"Error in optimization: {e}")
    
    def update_results(self, result):
        try:
            optimal_Q = result.x
            total_cost_value = result.fun
            
            # Calculate cost components
            order_cost = np.sum((self.D/optimal_Q) * self.O)
            hold_cost = np.sum((optimal_Q/2 + (self.D/12)*0.3) * self.H)
            penalty_cost = np.sum(np.maximum(0, self.D - optimal_Q)) * self.penalty_factor
            
            # Update result text
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Optimal Order Quantities:\n")
            for item, q in zip(self.items, optimal_Q):
                self.result_text.insert(tk.END, f"{item}: {q:.0f} units\n")
            
            # Update cost labels
            self.cost_labels['order'].config(text=f"Ordering Cost: Rs. {order_cost:.2f}")
            self.cost_labels['hold'].config(text=f"Holding Cost: Rs. {hold_cost:.2f}")
            self.cost_labels['penalty'].config(text=f"Penalty Cost: Rs. {penalty_cost:.2f}")
            self.cost_labels['total'].config(text=f"Total Cost: Rs. {total_cost_value:.2f}")
        except Exception as e:
            print(f"Error updating results: {e}")
    
    def update_visualization(self, result):
        try:
            self.ax.clear()
            
            # Create bar chart of optimal quantities
            optimal_Q = result.x
            self.ax.bar(self.items, optimal_Q)
            self.ax.set_title("Optimal Order Quantities")
            self.ax.set_ylabel("Quantity")
            self.ax.tick_params(axis='x', rotation=45)
            
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating visualization: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = InventoryOptimizationGUI(root)
    root.mainloop()
