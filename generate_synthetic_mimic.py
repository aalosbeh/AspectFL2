"""
Synthetic MIMIC-III-like Dataset Generator for Mortality Prediction
Generates realistic ICU patient data with 17 clinical variables
"""

import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_mimic_data(n_patients=5000, n_sites=5, mortality_rate=0.12):
    """
    Generate synthetic MIMIC-III-like dataset with realistic clinical variables
    """
    
    # Calculate patients per site
    base_per_site = n_patients // n_sites
    site_sizes = []
    for i in range(n_sites):
        if i < n_sites - 1:
            size = base_per_site + np.random.randint(-100, 100)
        else:
            size = n_patients - sum(site_sizes)
        site_sizes.append(max(size, 100))
    
    all_sites_data = {}
    
    for site_id in range(n_sites):
        n_site_patients = site_sizes[site_id]
        
        # Generate mortality labels
        n_deaths = int(n_site_patients * mortality_rate)
        mortality = np.array([1] * n_deaths + [0] * (n_site_patients - n_deaths))
        np.random.shuffle(mortality)
        
        # Generate all variables
        data = {
            'patient_id': [f'SITE{site_id}_P{i:05d}' for i in range(n_site_patients)],
            'age': [],
            'gender': [],
            'heart_rate': [],
            'systolic_bp': [],
            'diastolic_bp': [],
            'mean_arterial_pressure': [],
            'respiratory_rate': [],
            'temperature': [],
            'spo2': [],
            'gcs': [],
            'wbc': [],
            'hemoglobin': [],
            'platelet': [],
            'creatinine': [],
            'bun': [],
            'glucose': [],
            'sofa_score': [],
            'mortality': mortality
        }
        
        for i in range(n_site_patients):
            is_dead = mortality[i] == 1
            
            # Age
            if is_dead:
                age = np.clip(np.random.gamma(10, 8) + 18, 18, 95)
            else:
                age = np.clip(np.random.gamma(8, 8) + 18, 18, 95)
            data['age'].append(age)
            
            # Gender
            data['gender'].append(np.random.binomial(1, 0.55))
            
            # Heart Rate
            if is_dead:
                hr = np.clip(np.random.normal(95, 20), 40, 180)
            else:
                hr = np.clip(np.random.normal(80, 15), 40, 180)
            data['heart_rate'].append(hr)
            
            # Systolic BP
            if is_dead:
                sbp = np.clip(np.random.normal(105, 25), 60, 200)
            else:
                sbp = np.clip(np.random.normal(120, 20), 60, 200)
            data['systolic_bp'].append(sbp)
            
            # Diastolic BP
            if is_dead:
                dbp = np.clip(np.random.normal(60, 15), 30, 120)
            else:
                dbp = np.clip(np.random.normal(70, 12), 30, 120)
            data['diastolic_bp'].append(dbp)
            
            # Mean Arterial Pressure
            map_val = (sbp + 2 * dbp) / 3
            data['mean_arterial_pressure'].append(map_val)
            
            # Respiratory Rate
            if is_dead:
                rr = np.clip(np.random.normal(24, 6), 8, 40)
            else:
                rr = np.clip(np.random.normal(18, 4), 8, 40)
            data['respiratory_rate'].append(rr)
            
            # Temperature
            if is_dead:
                if np.random.rand() < 0.5:
                    temp = np.clip(np.random.normal(36.2, 0.8), 35.0, 40.5)
                else:
                    temp = np.clip(np.random.normal(38.5, 1.0), 35.0, 40.5)
            else:
                temp = np.clip(np.random.normal(37.0, 0.5), 35.0, 40.5)
            data['temperature'].append(temp)
            
            # SpO2
            if is_dead:
                spo2 = np.clip(np.random.normal(92, 5), 70, 100)
            else:
                spo2 = np.clip(np.random.normal(97, 2), 70, 100)
            data['spo2'].append(spo2)
            
            # GCS
            if is_dead:
                gcs = np.random.choice([3, 6, 9, 12, 15], p=[0.2, 0.3, 0.3, 0.15, 0.05])
            else:
                gcs = np.random.choice([15, 14, 13], p=[0.7, 0.2, 0.1])
            data['gcs'].append(gcs)
            
            # WBC
            if is_dead:
                wbc = np.clip(np.random.gamma(3, 4), 1, 30)
            else:
                wbc = np.clip(np.random.gamma(2.5, 3), 1, 30)
            data['wbc'].append(wbc)
            
            # Hemoglobin
            if is_dead:
                hgb = np.clip(np.random.normal(10, 2), 5, 18)
            else:
                hgb = np.clip(np.random.normal(12.5, 1.5), 5, 18)
            data['hemoglobin'].append(hgb)
            
            # Platelet
            if is_dead:
                plt = np.clip(np.random.gamma(4, 40), 20, 500)
            else:
                plt = np.clip(np.random.gamma(6, 40), 20, 500)
            data['platelet'].append(plt)
            
            # Creatinine
            if is_dead:
                creat = np.clip(np.random.gamma(3, 1), 0.5, 10)
            else:
                creat = np.clip(np.random.gamma(1.5, 0.5), 0.5, 10)
            data['creatinine'].append(creat)
            
            # BUN
            if is_dead:
                bun = np.clip(np.random.gamma(5, 8), 5, 100)
            else:
                bun = np.clip(np.random.gamma(3, 5), 5, 100)
            data['bun'].append(bun)
            
            # Glucose
            if is_dead:
                glucose = np.clip(np.random.gamma(8, 20), 50, 400)
            else:
                glucose = np.clip(np.random.gamma(6, 15), 50, 400)
            data['glucose'].append(glucose)
            
            # SOFA Score
            if is_dead:
                sofa = np.random.choice([4, 6, 8, 10, 12, 15, 18], p=[0.05, 0.1, 0.2, 0.25, 0.2, 0.15, 0.05])
            else:
                sofa = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.07, 0.03])
            data['sofa_score'].append(sofa)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Introduce missing data
        missing_rates = {
            'heart_rate': 0.05,
            'systolic_bp': 0.08,
            'diastolic_bp': 0.08,
            'mean_arterial_pressure': 0.08,
            'respiratory_rate': 0.06,
            'temperature': 0.10,
            'spo2': 0.07,
            'gcs': 0.12,
            'wbc': 0.15,
            'hemoglobin': 0.15,
            'platelet': 0.15,
            'creatinine': 0.18,
            'bun': 0.18,
            'glucose': 0.12,
            'sofa_score': 0.10
        }
        
        for col, miss_rate in missing_rates.items():
            if miss_rate > 0:
                n_missing = int(len(df) * miss_rate)
                missing_idx = np.random.choice(df.index, n_missing, replace=False)
                df.loc[missing_idx, col] = np.nan
        
        all_sites_data[f'site_{site_id}'] = df
    
    return all_sites_data

def save_datasets(data_dict, output_dir='data'):
    """Save datasets to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for site_name, df in data_dict.items():
        filepath = os.path.join(output_dir, f'{site_name}.csv')
        df.to_csv(filepath, index=False)
        print(f"Saved {site_name}: {len(df)} patients, mortality rate: {df['mortality'].mean():.2%}")
    
    # Save combined dataset
    combined_df = pd.concat(data_dict.values(), ignore_index=True)
    combined_path = os.path.join(output_dir, 'combined_all_sites.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"\nSaved combined dataset: {len(combined_df)} patients")
    
    # Generate data summary
    summary = {
        'total_patients': len(combined_df),
        'total_deaths': int(combined_df['mortality'].sum()),
        'mortality_rate': combined_df['mortality'].mean(),
        'n_sites': len(data_dict),
        'variables': list(combined_df.columns),
        'missing_data_summary': combined_df.isnull().sum().to_dict()
    }
    
    return summary

if __name__ == '__main__':
    print("Generating Synthetic MIMIC-III-like Dataset...")
    print("=" * 60)
    
    # Generate data for 5 sites with total 5000 patients
    data = generate_synthetic_mimic_data(n_patients=5000, n_sites=5, mortality_rate=0.12)
    
    # Save datasets
    summary = save_datasets(data, output_dir='data')
    
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print(f"Total Patients: {summary['total_patients']}")
    print(f"Total Deaths: {summary['total_deaths']}")
    print(f"Overall Mortality Rate: {summary['mortality_rate']:.2%}")
    print(f"Number of Sites: {summary['n_sites']}")
    print(f"\nVariables: {len(summary['variables'])}")
    print("\nMissing Data Summary:")
    for var, count in summary['missing_data_summary'].items():
        if count > 0:
            pct = (count / summary['total_patients']) * 100
            print(f"  {var}: {count} ({pct:.1f}%)")

