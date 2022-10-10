# Results comparison
Each segment contains 10s (0-9) trajectory datapoints. We compared using different length of input sequence:
- 3s (2-4)
- 4s (1-4)
- 5s (0-4)

to predict 5s (5-9) trajectory into future.
<p align="center">
  <img src="https://github.com/xichennn/LSTM-on-simulated-CARLA-data/blob/stage1_comparison/Figs/result_sub_1.jpg" width="300" title="q1">
  <img src="https://github.com/xichennn/LSTM-on-simulated-CARLA-data/blob/stage1_comparison/Figs/result_sub_2.jpg" width="300" title="q2">
  <img src="https://github.com/xichennn/LSTM-on-simulated-CARLA-data/blob/stage1_comparison/Figs/result_sub_3.jpg" width="300" alt="accessibility text">
</p>
Observation: Larger variance among predictions with lane change <br/>

5s input sequence gives the best performance. Hence, we compared adding extra features (vel_x and vel_y) into 5s input sequence.
<p align="center">
  <img src="https://github.com/xichennn/LSTM-on-simulated-CARLA-data/blob/stage1_comparison/Figs/rmse_compr.jpg" width="350" title="rmse">
</p>

Observations:
- Longer input sequence and more input features yield better performance
- Error values increase when the prediction horizon increases

