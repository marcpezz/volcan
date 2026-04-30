"""
Generate parameter combinations for 6 volcanic intrusion scenarios.
Uses Latin Hypercube Sampling for efficient parameter space exploration.
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import qmc
import pandas as pd

# Physical constants
KM = 1e3  # meters
YEAR = 3600 * 24 * 365.25  # seconds

class ScenarioGenerator:
    """Generate scenarios for volcanic thermal simulations."""
    
    def __init__(self, n_samples_per_scenario=100, seed=42):
        """
        Initialize scenario generator.
        
        Parameters:
        -----------
        n_samples_per_scenario : int
            Number of samples per scenario (100 * 10 = 1000 total)
        seed : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples_per_scenario
        self.seed = seed
        self.scenarios = []
        
    def define_parameter_ranges(self, scenario_id):
        """
        Define parameter ranges for each scenario.
        
        Returns dict with parameter bounds: {param_name: (min, max)}
        """
        # Common parameters across all scenarios
        common_params = {
            'magma_temp_C': (900, 1200),           # Magma temperature [°C]
            'injection_interval_yr': (500, 50000),  # Time between injections [years]
            'dike_thickness_m': (10, 500),          # Dike/sill thickness [m]
            'intrusion_width_km': (1, 20),          # Lateral extent [km]
            'host_conductivity': (2.0, 4.0),        # Thermal conductivity [W/m/K]
            'host_heat_capacity': (800, 1200),      # Heat capacity [J/kg/K]
            'geotherm_gradient': (20, 25),          # Geothermal gradient [K/km] - realistic range
            'simulation_duration_Myr': (1.0, 1.0),  # Fixed at 1.0 Myr for all scenarios
        }
        
        # Scenario-specific parameters
        scenario_params = {
            'S1': {  # Deep Dike Intrusions (lower crust)
                'depth_center_km': (20, 30),
                'flux_rate_km3_yr': (0.001, 0.05),
                'injection_pattern': 'steady',
                'depth_window': 'lower_crust',
            },
            'S2': {  # Mid-Crustal Dike Intrusions
                'depth_center_km': (10, 20),
                'flux_rate_km3_yr': (0.001, 0.03),
                'injection_pattern': 'steady',
                'depth_window': 'upper_crust',
            },
            'S3': {  # Shallow Dike Intrusions (upper crust)
                'depth_center_km': (5, 15),
                'flux_rate_km3_yr': (0.001, 0.03),
                'injection_pattern': 'steady',
                'depth_window': 'upper_crust',
            },
            'S4': {  # Transcrustal Dike System
                'depth_deep_km': (20, 30),
                'depth_shallow_km': (5, 10),
                'flux_deep_km3_yr': (0.005, 0.05),
                'flux_shallow_km3_yr': (0.001, 0.02),
                'flux_ratio': (0.2, 0.8),  # Shallow/deep flux ratio
                'injection_pattern': 'transcrustal',
                'depth_window': 'transcrustal',
            },
            'S5': {  # Two-Stage System (Deep → Shallow)
                'depth_stage1_km': (20, 30),
                'depth_stage2_km': (5, 15),
                'flux_stage1_km3_yr': (0.005, 0.05),
                'flux_stage2_km3_yr': (0.001, 0.02),
                'transition_time_Myr': (0.2, 0.8),
                'injection_pattern': 'two_stage',
                'depth_window': 'two_stage',
            },
            'S6': {  # GENEVA Lower Crust (radial accretion)
                'depth_center_km': (25, 25),  # Fixed at 25 km
                'sill_diameter_km': (10, 30),
                'sill_thickness_m': (50, 100),
                'flux_rate_km3_yr_km2': (7e-6, 15e-6),
                'injection_pattern': 'radial_accretion',
                'depth_window': 'lower_crust',
            },
            'S7': {  # GENEVA Upper Crust (radial accretion)
                'depth_center_km': (8, 8),  # Fixed at 8 km
                'sill_diameter_km': (5, 15),
                'sill_thickness_m': (30, 80),
                'flux_rate_km3_yr_km2': (10e-6, 20e-6),
                'injection_pattern': 'radial_accretion',
                'depth_window': 'upper_crust',
            },
            'S8': {  # UCLA Lower Crust (elliptical sills)
                'depth_center_km': (25, 25),  # Fixed at 25 km
                'sill_semi_major_km': (2, 5),
                'sill_aspect_ratio': (2, 5),  # a/b ratio
                'sill_volume_km3': (5, 20),
                'injection_pattern': 'elliptical_sills',
                'depth_window': 'lower_crust',
            },
            'S9': {  # UCLA Upper Crust (elliptical sills)
                'depth_center_km': (8, 8),  # Fixed at 8 km
                'sill_semi_major_km': (1, 3),
                'sill_aspect_ratio': (1.5, 3),  # a/b ratio
                'sill_volume_km3': (2, 10),
                'injection_pattern': 'elliptical_sills',
                'depth_window': 'upper_crust',
            },
            'S10': {  # Episodic Pulses (mid-crust)
                'depth_center_km': (10, 20),
                'active_duration_kyr': (50, 100),
                'quiet_duration_kyr': (100, 300),
                'n_pulses': (3, 5),
                'injection_pattern': 'episodic_pulses',
                'depth_window': 'mid_crust',
            },
        }
        
        # Merge common and scenario-specific parameters
        params = {**common_params}
        if scenario_id in scenario_params:
            params.update(scenario_params[scenario_id])
        
        return params, scenario_params[scenario_id]
    
    def latin_hypercube_sample(self, param_ranges):
        """
        Generate Latin Hypercube samples for parameter ranges.
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary of {param_name: (min, max)}
            
        Returns:
        --------
        samples : dict
            Dictionary of {param_name: array of samples}
        """
        # Extract numeric parameters only
        numeric_params = {k: v for k, v in param_ranges.items() 
                         if isinstance(v, tuple) and len(v) == 2}
        
        n_params = len(numeric_params)
        param_names = list(numeric_params.keys())
        
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=n_params, seed=self.seed)
        samples_unit = sampler.random(n=self.n_samples)
        
        # Scale to actual parameter ranges
        samples = {}
        for i, param_name in enumerate(param_names):
            min_val, max_val = numeric_params[param_name]
            if min_val == max_val:
                # Constant parameter - set all samples to this value
                samples[param_name] = np.full(self.n_samples, min_val)
            else:
                # Variable parameter - use LHS
                samples[param_name] = qmc.scale(
                    samples_unit[:, i:i+1], 
                    [min_val], 
                    [max_val]
                ).flatten()
        
        return samples
    
    def generate_scenario(self, scenario_id, scenario_name, references):
        """
        Generate all parameter combinations for a single scenario.
        
        Parameters:
        -----------
        scenario_id : str
            Scenario identifier (S1-S6)
        scenario_name : str
            Human-readable scenario name
        references : list
            List of reference papers
            
        Returns:
        --------
        scenarios : list of dict
            List of scenario parameter dictionaries
        """
        param_ranges, scenario_specific = self.define_parameter_ranges(scenario_id)
        samples = self.latin_hypercube_sample(param_ranges)
        
        scenarios = []
        for i in range(self.n_samples):
            scenario = {
                'scenario_id': scenario_id,
                'scenario_name': scenario_name,
                'sample_id': i,
                'global_id': len(self.scenarios) + i,
                'references': references,
            }
            
            # Add all sampled parameters
            for param_name, values in samples.items():
                scenario[param_name] = float(values[i])
            
            # Add categorical parameters
            for key, value in scenario_specific.items():
                if not isinstance(value, tuple):
                    scenario[key] = value
            
            # Compute derived parameters
            scenario = self._compute_derived_params(scenario, scenario_id)
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _compute_derived_params(self, scenario, scenario_id):
        """Compute derived parameters for simulation setup."""
        
        # Convert units for Julia simulation
        scenario['magma_temp_K'] = scenario['magma_temp_C'] + 273.15
        scenario['injection_interval_s'] = scenario['injection_interval_yr'] * YEAR
        scenario['dike_thickness'] = scenario['dike_thickness_m']
        scenario['intrusion_width'] = scenario['intrusion_width_km'] * KM
        scenario['simulation_duration_s'] = scenario['simulation_duration_Myr'] * 1e6 * YEAR
        
        # Compute domain size based on scenario type
        # Fixed domain for all scenarios as per documentation
        scenario['domain_depth_km'] = 30  # Fixed 30 km depth
        scenario['domain_width_km'] = 40  # Fixed 40 km width
        
        # Grid resolution - fixed for all scenarios
        # 200m resolution for faster, lighter simulations
        min_resolution = 200  # meters
        scenario['grid_nx'] = int(scenario['domain_width_km'] * KM / min_resolution)
        scenario['grid_nz'] = int(scenario['domain_depth_km'] * KM / min_resolution)
        
        # This gives 200 x 150 grid for 40km x 30km domain at 200m resolution (30,000 cells)
        
        return scenario
    
    def generate_all_scenarios(self):
        """Generate all 10 scenarios with their parameter combinations."""
        
        scenario_definitions = [
            ('S1', 'Deep Dike Intrusions (Lower Crust)', 
             ['Annen & Sparks 2002', 'Annen et al. 2006']),
            ('S2', 'Mid-Crustal Dike Intrusions', 
             ['Annen et al. 2009', 'Glazner et al. 2021']),
            ('S3', 'Shallow Dike Intrusions (Upper Crust)', 
             ['Annen et al. 2006', 'Cashman et al. 2017']),
            ('S4', 'Transcrustal Dike System', 
             ['Cashman et al. 2017', 'Liu & Lee 2020']),
            ('S5', 'Two-Stage System (Deep → Shallow)', 
             ['Caricchi et al. 2014', 'Liu & Lee 2020']),
            ('S6', 'GENEVA Lower Crust (Radial Accretion)', 
             ['Annen et al. 2015', 'Karakas et al. 2017']),
            ('S7', 'GENEVA Upper Crust (Radial Accretion)', 
             ['Annen et al. 2015', 'Karakas et al. 2017']),
            ('S8', 'UCLA Lower Crust (Elliptical Sills)', 
             ['Dufek & Bergantz 2005', 'Annen et al. 2015']),
            ('S9', 'UCLA Upper Crust (Elliptical Sills)', 
             ['Dufek & Bergantz 2005', 'Annen et al. 2015']),
            ('S10', 'Episodic Pulses (Mid-Crust)', 
             ['Caricchi et al. 2014', 'Barboni et al. 2016']),
        ]
        
        all_scenarios = []
        for scenario_id, name, refs in scenario_definitions:
            print(f"Generating {name} ({scenario_id})...")
            scenarios = self.generate_scenario(scenario_id, name, refs)
            all_scenarios.extend(scenarios)
            self.scenarios.extend(scenarios)
        
        print(f"\nTotal scenarios generated: {len(all_scenarios)}")
        return all_scenarios
    
    def save_scenarios(self, output_dir='scenarios'):
        """Save scenarios to individual JSON files and summary CSV."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save individual JSON files
        for scenario in self.scenarios:
            filename = f"scenario_{scenario['global_id']:04d}_{scenario['scenario_id']}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(scenario, f, indent=2)
        
        # Save summary CSV
        df = pd.DataFrame(self.scenarios)
        summary_file = output_path / 'scenarios_summary.csv'
        df.to_csv(summary_file, index=False)
        
        print(f"\nSaved {len(self.scenarios)} scenarios to {output_dir}/")
        print(f"Summary saved to {summary_file}")
        
        # Print statistics
        print("\nScenario distribution:")
        print(df['scenario_id'].value_counts().sort_index())
        
        return df


def main():
    """Main execution function."""
    
    # Generate scenarios
    generator = ScenarioGenerator(n_samples_per_scenario=100, seed=42)
    scenarios = generator.generate_all_scenarios()
    
    # Save to disk
    df = generator.save_scenarios(output_dir='scenarios')
    
    # Print example scenario
    print("\n" + "="*60)
    print("Example scenario (S1, sample 0):")
    print("="*60)
    example = [s for s in scenarios if s['scenario_id'] == 'S1'][0]
    for key, value in example.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4e}")
        else:
            print(f"{key:30s}: {value}")


if __name__ == '__main__':
    main()
