import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

# Configure logging
logging.basicConfig(
    filename='inventory_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class InventoryOptimizer:
    def __init__(self, data_file: str = None):
        """Initialize the inventory optimizer with data from file or default values"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing InventoryOptimizer")
        
        if data_file and os.path.exists(data_file):
            try:
                self.load_data_from_file(data_file)
                self.logger.info(f"Successfully loaded data from {data_file}")
            except Exception as e:
                self.logger.error(f"Error loading data from file: {e}")
                self.load_default_data()
        else:
            self.load_default_data()
        
        # Initialize seasonal parameters
        self.festive_months = [10, 11, 12]  # October, November, December
        self.festive_multiplier = 1.5
        
        # Initialize visualization
        plt.style.use('ggplot')
    
    def load_default_data(self):
        """Load default data if no file is provided"""
        self.logger.info("Loading default data")
        self.items = ["Bed", "Sofa", "Table", "Office Chair", "Almirah", "Stool", "Shoe Rack", 
                     "Dressing Table", "Dining Table", "Study Table", "Pillow", "Mattress", 
                     "Bedsheet", "Wooden Mandir", "Plastic Chair"]
        
        self.base_D = np.array([72, 84, 120, 120, 96, 120, 84, 60, 24, 84, 240, 300, 60, 36, 1200])
        self.O = np.array([500, 500, 400, 300, 400, 200, 200, 300, 400, 300, 100, 500, 100, 200, 150])
        self.H = np.array([300, 400, 250, 200, 250, 100, 100, 200, 250, 200, 50, 400, 50, 150, 75])
        self.V = np.array([50, 70, 60, 40, 30, 20, 30, 40, 50, 40, 10, 20, 10, 30, 10])
        self.L = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.MOQ = np.array([2, 2, 2, 3, 1, 3, 3, 2, 2, 2, 5, 2, 5, 2, 10])
        self.importance_factors = np.array([1.2, 1.2, 1.1, 1.0, 1.3, 1.0, 1.0, 1.1, 1.2, 1.0, 1.0, 1.3, 1.0, 1.2, 1.0])
    
    def load_data_from_file(self, file_path: str) -> None:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            required_columns = ['item', 'base_demand', 'ordering_cost', 'holding_cost', 
                              'volume', 'lead_time', 'moq', 'importance_factor']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in data file")
            
            self.items = df['item'].tolist()
            self.base_D = df['base_demand'].values
            self.O = df['ordering_cost'].values
            self.H = df['holding_cost'].values
            self.V = df['volume'].values
            self.L = df['lead_time'].values
            self.MOQ = df['moq'].values
            self.importance_factors = df['importance_factor'].values
            self.validate_data()
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self) -> None:
        """Validate loaded data for consistency and correctness"""
        try:
            for name, array in [('base_demand', self.base_D), ('ordering_cost', self.O),
                              ('holding_cost', self.H), ('volume', self.V),
                              ('lead_time', self.L), ('moq', self.MOQ)]:
                if np.any(array < 0):
                    raise ValueError(f"Negative values found in {name}")
            
            lengths = [len(self.items), len(self.base_D), len(self.O), len(self.H),
                      len(self.V), len(self.L), len(self.MOQ), len(self.importance_factors)]
            if len(set(lengths)) != 1:
                raise ValueError("Inconsistent array lengths in data")
                
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise
    
    def get_seasonal_demand(self, base_demand: np.ndarray) -> np.ndarray:
        """Calculate demand adjusted for festive season with different multipliers for different items"""
        current_month = datetime.now().month
        
        # Define seasonal multipliers for different items based on Indian market trends
        seasonal_multipliers = {
            "Bed": 1.3,  # Higher demand during wedding season
            "Sofa": 1.4,  # Popular during festive season
            "Table": 1.2,  # Moderate increase
            "Office Chair": 1.1,  # Slight increase
            "Almirah": 1.3,  # Higher demand during festive season
            "Stool": 1.2,  # Moderate increase
            "Shoe Rack": 1.3,  # Higher demand during festive season
            "Dressing Table": 1.4,  # Popular during wedding season
            "Dining Table": 1.5,  # Highest increase during festive season
            "Study Table": 1.2,  # Moderate increase
            "Pillow": 1.3,  # Higher demand during festive season
            "Mattress": 1.4,  # Popular during wedding season
            "Bedsheet": 1.3,  # Higher demand during festive season
            "Wooden Mandir": 1.5,  # Highest increase during festive season
            "Plastic Chair": 1.2  # Moderate increase
        }
        
        # Apply multipliers based on current month
        if current_month in [10, 11, 12]:  # Festive season (Diwali, Christmas)
            multipliers = np.array([seasonal_multipliers[item] for item in self.items])
            seasonal_demand = base_demand * multipliers
        elif current_month in [4, 5, 6]:  # Wedding season
            multipliers = np.array([seasonal_multipliers[item] for item in self.items])
            seasonal_demand = base_demand * multipliers
        else:
            seasonal_demand = base_demand * 1.1  # Base increase of 10% during other months
        
        # Round to nearest integers
        seasonal_demand = np.round(seasonal_demand)
        
        # Log the seasonal adjustments
        self.logger.info(f"Applying seasonal multipliers for month {current_month}")
        for item, base, seasonal in zip(self.items, base_demand, seasonal_demand):
            if base != seasonal:
                self.logger.info(f"{item}: Base demand {base} -> Seasonal demand {seasonal}")
        
        return seasonal_demand
    
    def calculate_safety_stock(self, daily_demand: np.ndarray, lead_time: np.ndarray) -> np.ndarray:
        """Calculate safety stock based on service level and demand variability"""
        z_score = 1.645  # For 95% service level
        demand_std = daily_demand * 0.2  # Assume 20% demand variability
        return z_score * demand_std * np.sqrt(lead_time)
    
    def calculate_reorder_point(self, daily_demand: np.ndarray, safety_stock: np.ndarray) -> np.ndarray:
        """Calculate reorder point based on lead time demand and safety stock"""
        return daily_demand * self.L + safety_stock
    
    def calculate_understocking_penalty(self, understock_units: np.ndarray) -> np.ndarray:
        """Calculate penalty cost for understocking with more realistic penalties"""
        try:
            # Base penalty is 2% of item value per unit
            base_unit_penalty = self.H * 2  # Using holding cost as proxy for item value
            
            # Progressive penalty structure
            penalty_multiplier = np.where(
                understock_units > 20, 3.0,  # Severe understocking
                np.where(understock_units > 10, 2.0,  # Moderate understocking
                np.where(understock_units > 5, 1.5,  # Mild understocking
                1.0)))  # Base penalty
            
            # Calculate total penalty with importance factor
            total_penalty = base_unit_penalty * understock_units * penalty_multiplier * self.importance_factors
            
            # Add minimum penalty for any understocking
            total_penalty = np.where(understock_units > 0, 
                                   np.maximum(total_penalty, self.O * 0.1),  # At least 10% of ordering cost
                                   total_penalty)
            
            # Log significant penalties
            for i, (item, penalty) in enumerate(zip(self.items, total_penalty)):
                if penalty > 0:
                    self.logger.info(f"Understocking penalty for {item}: Rs. {penalty:,.2f}")
            
            return total_penalty
            
        except Exception as e:
            self.logger.error(f"Error calculating understocking penalty: {e}")
            raise
    
    def total_cost(self, Q: np.ndarray) -> float:
        """Calculate total cost including ordering, holding, and penalty costs"""
        if np.any(Q <= 0):
            return np.inf
        
        Q = np.maximum(Q, self.MOQ)
        current_D = self.get_seasonal_demand(self.base_D)
        daily_demand = current_D / 365
        safety_stock = self.calculate_safety_stock(daily_demand, self.L)
        avg_inventory = Q/2 + safety_stock
        
        order_cost = (current_D / Q) * self.O
        hold_cost = avg_inventory * self.H
        understock_units = np.maximum(0, safety_stock - Q/2)
        understock_penalty = self.calculate_understocking_penalty(understock_units)
        
        return np.sum(order_cost + hold_cost + understock_penalty)
    
    def storage_constraint(self, Q: np.ndarray) -> float:
        """Storage capacity constraint"""
        return 2000 - np.sum(Q * self.V)
    
    def demand_constraint(self, Q: np.ndarray) -> np.ndarray:
        """Demand constraint"""
        current_D = self.get_seasonal_demand(self.base_D)
        return Q - current_D
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Run the optimization process"""
        constraints = [{'type': 'ineq', 'fun': self.storage_constraint},
                       {'type': 'ineq', 'fun': self.demand_constraint}]
        bounds = [(m, None) for m in self.MOQ]
        Q0 = self.MOQ * 2
        
        result = minimize(self.total_cost, Q0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            self.logger.warning(f"Optimization did not converge: {result.message}")
        
        return result.x, result.fun

    def perform_sensitivity_analysis(self, optimal_Q: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Perform sensitivity analysis on key parameters"""
        try:
            self.logger.info("Starting sensitivity analysis")
            results = {}
            
            # Original values
            original_H = self.H.copy()
            original_O = self.O.copy()
            original_V = self.V.copy()
            
            # Test holding cost variations
            holding_costs = []
            for multiplier in [0.8, 1.2]:
                self.H = original_H * multiplier
                _, cost = self.optimize()
                holding_costs.append(cost)
            results['Holding Cost'] = (min(holding_costs), max(holding_costs))
            
            # Test ordering cost variations
            ordering_costs = []
            for multiplier in [0.8, 1.2]:
                self.O = original_O * multiplier
                _, cost = self.optimize()
                ordering_costs.append(cost)
            results['Ordering Cost'] = (min(ordering_costs), max(ordering_costs))
            
            # Test storage volume variations
            storage_costs = []
            for multiplier in [0.8, 1.2]:
                self.V = original_V * multiplier
                _, cost = self.optimize()
                storage_costs.append(cost)
            results['Storage Volume'] = (min(storage_costs), max(storage_costs))
            
            # Restore original values
            self.H = original_H
            self.O = original_O
            self.V = original_V
            
            # Log the results
            self.logger.info("Sensitivity Analysis Results:")
            for param, (min_cost, max_cost) in results.items():
                self.logger.info(f"{param}: Rs. {min_cost:,.2f} to Rs. {max_cost:,.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in sensitivity analysis: {e}")
            raise

    def generate_report(self, optimal_Q: np.ndarray, total_cost: float) -> str:
        """Generate a detailed report of the optimization results"""
        try:
            report = []
            report.append("Inventory Optimization Report")
            report.append("=" * 50)
            report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Summary section
            report.append("Summary")
            report.append("-" * 50)
            report.append(f"Total Cost: Rs. {total_cost:,.2f}")
            report.append(f"Total Storage Volume Used: {np.sum(optimal_Q * self.V):,.2f} units\n")
            
            # Item-wise details
            report.append("Item-wise Details")
            report.append("-" * 50)
            current_D = self.get_seasonal_demand(self.base_D)
            daily_demand = current_D / 365
            safety_stock = self.calculate_safety_stock(daily_demand, self.L)
            reorder_points = self.calculate_reorder_point(daily_demand, safety_stock)
            
            for i, item in enumerate(self.items):
                report.append(f"\n{item}:")
                report.append(f"  Base Demand: {self.base_D[i]:.0f} units/year")
                report.append(f"  Seasonal Demand: {current_D[i]:.0f} units/year")
                report.append(f"  Optimal Order Quantity: {optimal_Q[i]:.0f} units")
                report.append(f"  Reorder Point: {reorder_points[i]:.0f} units")
                report.append(f"  Safety Stock: {safety_stock[i]:.0f} units")
                report.append(f"  Storage Volume: {self.V[i] * optimal_Q[i]:.0f} units")
                report.append(f"  Ordering Cost: Rs. {(current_D[i] / optimal_Q[i]) * self.O[i]:,.2f}")
                report.append(f"  Holding Cost: Rs. {(optimal_Q[i]/2 + safety_stock[i]) * self.H[i]:,.2f}")
            
            # Cost breakdown
            report.append("\nCost Breakdown")
            report.append("-" * 50)
            order_costs = (current_D / optimal_Q) * self.O
            hold_costs = (optimal_Q/2 + safety_stock) * self.H
            understock_units = np.maximum(0, safety_stock - optimal_Q/2)
            penalty_costs = self.calculate_understocking_penalty(understock_units)
            
            report.append(f"Total Ordering Cost: Rs. {np.sum(order_costs):,.2f}")
            report.append(f"Total Holding Cost: Rs. {np.sum(hold_costs):,.2f}")
            report.append(f"Total Penalty Cost: Rs. {np.sum(penalty_costs):,.2f}")
            report.append(f"Total Cost: Rs. {total_cost:,.2f}")
            
            # Write report to file
            report_filename = 'inventory_optimization_report.txt'
            with open(report_filename, 'w') as file:
                file.write('\n'.join(report))
            
            self.logger.info(f"Report successfully generated and saved to {report_filename}")
            return '\n'.join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise

    def export_results_to_excel(self, optimal_Q: np.ndarray, total_cost: float, 
                              file_path: str = 'optimization_results.xlsx') -> None:
        """Export optimization results to Excel"""
        try:
            current_D = self.get_seasonal_demand(self.base_D)
            daily_demand = current_D / 365
            safety_stock = self.calculate_safety_stock(daily_demand, self.L)
            reorder_points = self.calculate_reorder_point(daily_demand, safety_stock)
            
            # Create detailed results DataFrame
            detailed_data = {
                'Item': self.items,
                'Base Demand': self.base_D,
                'Seasonal Demand': current_D,
                'Optimal Order Quantity': optimal_Q,
                'Reorder Point': reorder_points,
                'Safety Stock': safety_stock,
                'Storage Volume': self.V * optimal_Q,
                'Ordering Cost': (current_D / optimal_Q) * self.O,
                'Holding Cost': (optimal_Q/2 + safety_stock) * self.H
            }
            detailed_df = pd.DataFrame(detailed_data)
            
            # Create summary DataFrame
            summary_data = {
                'Metric': ['Total Cost', 'Total Storage Volume Used'],
                'Value': [total_cost, np.sum(optimal_Q * self.V)]
            }
            summary_df = pd.DataFrame(summary_data)
            
            # Write to Excel
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            self.logger.info(f"Results exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            raise

    def visualize_results(self, optimal_Q: np.ndarray, save_path: Optional[str] = None) -> None:
        """Visualize optimization results with both order quantities and cost breakdown"""
        try:
            # Set style
            plt.style.use('ggplot')
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Define colors
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', 
                     '#99CCFF', '#CC99FF', '#FFD700', '#98FB98', '#FFA07A',
                     '#20B2AA', '#FFB6C1', '#87CEEB', '#DDA0DD', '#F0E68C']
            
            # Calculate costs for the pie chart
            current_D = self.get_seasonal_demand(self.base_D)
            daily_demand = current_D / 365
            safety_stock = self.calculate_safety_stock(daily_demand, self.L)
            
            order_costs = (current_D / optimal_Q) * self.O
            hold_costs = (optimal_Q/2 + safety_stock) * self.H
            understock_units = np.maximum(0, safety_stock - optimal_Q/2)
            penalty_costs = self.calculate_understocking_penalty(understock_units)
            
            total_order_cost = np.sum(order_costs)
            total_hold_cost = np.sum(hold_costs)
            total_penalty_cost = np.sum(penalty_costs)
            
            # Plot 1: Optimal Order Quantities
            bars = ax1.bar(self.items, optimal_Q, color=colors, edgecolor='black', linewidth=1)
            ax1.set_title('Optimal Order Quantities', fontsize=16, fontweight='bold', pad=20, color='#333333')
            ax1.set_xlabel('Items', fontsize=14, fontweight='bold', color='#333333')
            ax1.set_ylabel('Quantity', fontsize=14, fontweight='bold', color='#333333')
            ax1.tick_params(axis='x', rotation=45, labelsize=12)
            ax1.tick_params(axis='y', labelsize=12)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            # Add grid
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot 2: Cost Breakdown
            cost_labels = ['Ordering Cost', 'Holding Cost', 'Penalty Cost']
            cost_values = [total_order_cost, total_hold_cost, total_penalty_cost]
            cost_colors = ['#FF9999', '#66B2FF', '#99FF99']
            
            wedges, texts, autotexts = ax2.pie(cost_values, labels=cost_labels, colors=cost_colors,
                                             autopct='%1.1f%%', startangle=90,
                                             textprops={'fontsize': 12, 'fontweight': 'bold'})
            
            ax2.set_title('Cost Breakdown', fontsize=16, fontweight='bold', pad=20, color='#333333')
            
            # Add absolute values to the pie chart
            for i, (wedge, value) in enumerate(zip(wedges, cost_values)):
                angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                x = 0.8 * np.cos(np.deg2rad(angle))
                y = 0.8 * np.sin(np.deg2rad(angle))
                ax2.text(x, y, f'Rs. {value:,.2f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='#333333')
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)  # Add some space between subplots
            
            # Add a title to the entire figure
            fig.suptitle('Inventory Optimization Results', fontsize=18, fontweight='bold', y=1.02, color='#333333')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
                self.logger.info(f"Saved visualization to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error in visualization: {e}")
            raise

    def main_with_report(self) -> None:
        """Run the optimization with detailed reporting"""
        try:
            # Run optimization
            optimal_Q, total_cost = self.optimize()
            
            # Print results
            print("\nOptimal Order Quantities:")
            for item, qty in zip(self.items, optimal_Q):
                print(f"{item}: {int(qty)} units")
            print(f"\nTotal Cost: Rs. {total_cost:,.2f}")
            
            # Generate report
            report = self.generate_report(optimal_Q, total_cost)
            print("\nDetailed Report:")
            print(report)
            
            # Perform sensitivity analysis
            print("\nPerforming Sensitivity Analysis...")
            sensitivity_results = self.perform_sensitivity_analysis(optimal_Q)
            print("\nSensitivity Analysis Results:")
            for param, (min_cost, max_cost) in sensitivity_results.items():
                print(f"{param}: Rs. {min_cost:,.2f} to Rs. {max_cost:,.2f}")
            
            # Visualize results
            self.visualize_results(optimal_Q, 'optimization_results.png')
            
            # Export results to Excel
            self.export_results_to_excel(optimal_Q, total_cost)
            
        except Exception as e:
            print(f"Error in main execution: {e}")
            raise

if __name__ == "__main__":
    # Use the data file if it exists, otherwise use default data
    data_file = 'inventory_data.csv'
    optimizer = InventoryOptimizer(data_file)
    optimizer.main_with_report()
