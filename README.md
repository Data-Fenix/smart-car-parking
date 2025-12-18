# Smart Car Parking: Predictive Availability System
![Cover Image](gif_ppt.gif)

A machine learning system that predicts parking spot availability using historical occupancy, weather, and location data. Achieves **91.6% R²** with **~1 spot MAE** using Random Forest regression.

## 🎯 Project Overview

This project forecasts available parking spots for city road segments to help:

- **Drivers**: Find parking faster and reduce search time
- **Operators**: Optimize capacity and enable dynamic pricing
- **Cities**: Reduce traffic congestion and emissions
- **Businesses**: Increase foot traffic through accessible parking

**Key Achievement:** Predictions are accurate within ~1 parking spot on average, making this suitable for real-time guidance systems.

---

## 📊 Project Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 1.04 spots | Average prediction error |
| **RMSE** | 2.11 spots | Typical error magnitude |
| **R² Score** | 0.916 | Explains 91.6% of variance |

**Dataset Scale:**

- 4 months of data
- 57 road segments
- 1.25M+ observations
- 13.4M parking slots tracked
- Time coverage: 06:00–21:00 daily

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `case_study_smart_car_parking_lahiru_dissanayake.ipynb` | Complete analysis notebook with code and visualizations |
| `case_study_smart_car_parking_lahiru_dissanayake.pdf` | Comprehensive case study report |
| `jupyter_notebook_of_case_study_smart_car_parking_lahiru_dissanayake.pdf` | Notebook exported to PDF format |
| `PPT.pdf` | Presentation slides summarizing key findings |
| `evaluation_metrics.png` | Model performance visualization |
| `cleaned_data.csv` | Processed and merged dataset |
| `groundtruth.csv` | Raw occupancy data |
| `weather_features.csv` | Weather-related features |
| `road_features.csv` | Road and location features |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook

### Installation

```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch the notebook
jupyter notebook case_study_smart_car_parking_lahiru_dissanayake.ipynb
```

---

## 🔍 Key Insights

### Temporal Patterns

- **Peak demand**: Weekdays 10:00–15:00 and 18:00–21:00
- **Growing trend**: Upward trajectory in occupancy over 4 months
- **Event impact**: Evening hours (18:00–21:00) show elevated occupancy

### Environmental Effects

- **Weather sensitivity**: Occupancy drops during extreme temperatures (<7°C or >30°C) and rainy conditions
- **Opportunity**: Weather-based pricing incentives to manage demand

### Location Factors

- **High-demand areas**: Roads near restaurants, shopping, and residential zones show 40%+ higher occupancy
- **Transit gap**: Only 25% of segments have public transport access—expansion opportunity for park-and-ride
- **Top predictors**: Max capacity, current occupancy, restaurant proximity, residential density

---

## 🤖 Model Details

### Approach

- **Algorithm**: Random Forest Regressor
- **Task**: Predict number of available parking spots (regression)
- **Features**: 20+ variables including temporal, weather, location, and engineered lag/rolling features

### Why Random Forest?

- Handles non-linear relationships and feature interactions
- Robust to outliers and missing data
- Provides interpretable feature importance
- Excellent performance with minimal tuning

### Feature Categories

- **Temporal**: Hour, day of week, month, weekend indicators, event time flags
- **Weather**: Temperature, wind speed, precipitation
- **Location**: Commercial/residential density, nearby facilities (restaurants, schools, offices, shopping, supermarkets), off-street parking capacity
- **Engineered**: Lag features (1, 3, 6 periods) and rolling averages on occupancy and capacity

---

## 💡 Business Applications

### Real-time Driver Guidance

- Mobile app showing predicted availability by location and time
- Navigation to spots with highest availability

### Dynamic Pricing

- Increase rates during peak hours (10:00–15:00, 18:00–21:00)
- Weather-based discounts during adverse conditions
- Event-based premium pricing

### Infrastructure Planning

- Identify high-demand segments needing capacity expansion
- Prioritize public transport integration in underserved areas
- Optimize placement of off-street facilities

### Demand Management

- Price incentives to shift demand to off-peak hours
- Promote alternative segments during peak times

---

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (Random Forest)
- **Visualization**: matplotlib, seaborn
- **Environment**: Jupyter Notebook

---

## 📈 Future Enhancements

### Data Enrichment

- Real-time event calendars (concerts, sports, festivals)
- Live traffic volume and congestion data
- Public transport schedules and delays
- Detailed weather forecasts (humidity, visibility)

### Model Improvements

- Hyperparameter optimization (grid/random search)
- Ensemble methods (XGBoost, LightGBM, stacking)
- Deep learning for temporal patterns (LSTM, GRU)
- Feature interactions and polynomial features

### Production Readiness

- REST API for real-time predictions
- Automated retraining pipeline
- A/B testing framework for pricing strategies
- Comprehensive monitoring dashboard

---

## 👤 Author

Lahiru Dissanayake

For complete methodology, analysis, and results, refer to the full case study PDF.

---

## 📄 License

MIT License - See file for details

---

## 📚 Documentation

- **Detailed Analysis**: Case Study PDF
- **Code Walkthrough**: Jupyter Notebook
- **Presentation**: PPT Slides
