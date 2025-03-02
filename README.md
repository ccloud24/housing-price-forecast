# housing-price-forecast

This project aims to predict house prices based on various features such as square footage, number of bedrooms, bathrooms, and location. A neural network model is used for prediction.

## Steps:
1. **Load Dataset**: A CSV containing data about houses and their prices.
2. **Preprocessing**: Categorical data is encoded, and features are scaled.
3. **Model Training**: A neural network is trained using TensorFlow/Keras.
4. **Prediction**: The trained model is used to predict house prices for new inputs.

## Dependencies:
- pandas
- numpy
- tensorflow
- scikit-learn

## How to Run:
1. Install dependencies: pip install -r requirements.txt
2. Run the main script: house price predictor.py
3. The model will output the predicted house price.
   - home parameters can be adjusted on lines 47-50

## Future Improvements:
- Tune model hyperparameters
- Add more features for better prediction accuracy
- Visualize data and results
   
