# Sim2Real-Fire

**Sim2Real-Fire: A Multi-modal Simulation Dataset for Forecast and Backtracking of Real-world Forest Fire**

<p align="center">
     <img src="assets/architecture.png" alt="Sim2Real-Fire" width="800px">
     <br/> S2R-FireTr forecasts wildfires by predicting future target fire areas based on source fire areas. (a) During the environment-guided area representation learning, we input the source fire areas and multi-modal environmental information into the transformer to compute the source area presentation A. (b) During the time-wise fire area regression, we input the source area presentation A and the target timestamp into the transformer to compute the target area presentation R for predicting the target fire areas. "Shifted Later" means that we concatenate the source and target areas to predict later areas. Source and target areas can be interchanged, creating a pipeline for wildfire backtracking.
</p>

## Getting Started

### Sim2Real-Fire Dataset

We create a brand-new **Sim2Real-Fire Dataset** dataset contains wildﬁre simulation and real-world data, including **1M** and **1K** wildﬁre scenarios. Each scenario is associated with five data modalities of environmental information, including topography, vegetation, fuel maps, weather data, and satellite images with the annotated fire regions.

We provide a **mini-version** of the dataset in [Sim2Real-Fire](https://1drv.ms/f/s!AhX2uIQNmngrafE5KFjNyZym_7o?e=yo3Q1K)

Below is the directory structure of the Sim2Real-Fire dataset:
<p align="left">
  <img src="assets/dir.png" alt="Sim2Real-Fire dataset" width="300px">
</p>

Here, we present some examples of satellite images with the real fire areas. The complete collection is available in the directory 1.5_Satellite_Images/original_images. 


<p align="center">
  <img src="assets/original_images.png" alt="Sim2Real-Fire dataset" width="800px">
  <br/> Examples of satellite images with the real fire areas.
</p>

We will provide the full dataset upon the paper acceptance.

<br>

<p align="center">
  <img src="assets/examples.png" alt="Sim2Real-Fire dataset" width="800px">
  <br/> Topography, vegetation, fuel, weather, and the satellite data in the Sim2Real-Fire dataset.
</p>

<br>

<p align="center">
     <img src="assets/pie.png" alt="Sim2Real-Fire dataset distribution" width="800px">
     <br/> (a) Distribution of vegetation covers and types. (b) Distribution of fuel types. (c) Distribution topography data. (d) Distribution of weather data.
</p>


### Install dependencies
To set up the environment, install the dependencies using the following command:
```sh
pip install -r requirements.txt
```

### Training

To train the S2R-FireTr model on the Sim2Real-Fire simulation data:

1. **Configure the Training Parameters:**
   - Open the `config.yaml` file and adjust the training parameters, including learning rate, batch size, number of epochs, and dataset paths.
   - **Important:** Make sure to update the paths to the simulation data and real-world data in the `config.yaml` file according to your local file structure.

2. **Start Training:**
   - Run the training script with the following command:
   ```sh
   python train.py --config config.yaml
   ```
   - This will start the training process using the simulation data defined in the configuration file. During training, checkpoints will be saved periodically, and training logs will be outputted for monitoring.

### Testing

To test the trained S2R-FireTr model on the Sim2Real-Fire real-world data:

1. **Configure the Testing Parameters:**
   - Adjust the testing configuration in `config.yaml`, including paths to the trained model checkpoint and testing data.
   - **Important:** Ensure that the dataset paths in the `config.yaml` file point to the correct location of the test dataset on your system.

2. **Run the Testing Script:**
   - Use the following command to start testing:
   ```sh
   python test.py --config config.yaml
   ```
   - The script will load the trained model and evaluate it on the specified real-world data. The results, including performance metrics and visualizations. These visualizations, along with the evaluation metrics, will be stored in the designated output directory for further analysis and presentation.

3. **Evaluate Model Performance:**
   - After testing, review the output directory to analyze both the quantitative metrics and the qualitative visualizations to assess the model’s performance on real-world fire data.
