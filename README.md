# Project Title

In the past recent years, SLAM has increase it popularity, since the demand of autonomous driving is increasing. SLAM problem is a direct problem that is related to the Autonomous vehicle. It's the back bones behind unmanned vehicles and drones, self driving cars, robotics and also augmented reality applications.\\
\indent In this project we introduce a technique called Kalman Filter, which is close to state of the art technique in SLAM. The assumption is that our model comes from a Gaussian Distribution. Through an iterative method of update and prediction, we hope to obtain the best approximation of the landmarks position and the robot pose as the same time perform mapping.
## Problem Formulation
The SLAM bot in this project is a simple robot that is similar to a car that can move around and rotate. When the robot move around the environment we need to perform two tasks: Localization and Mapping. Both are using the idea of Kalman filter.  \\
\indent To perform visual mapping, our assumption is that the inverse IMU pose over time is known, and also the data association over time of the landmarks observed at each time is also known and static. Then our objective is that given the visual feature of the observations, we would like to estimate the homogeneous coordinates in the world frame of the landmarks. From the predicted and updated landmarks, we wish to update our pose of the robot based on the landmarks position.
### Technical Approach

As mentioned in the introduction section, our project targeting two important tasks which is Mapping and localization. Given the reading of the IMU which consists of angular and linear velocity; we wish to make prediction of the robot pose(mean) and also the co variance matrix based on Extended Kalman filter idea of the assumption that our motion model come from a Gaussian Distribution. 



### Result

Here is the Trajectory; vitual features when the robot moving around and the features in 3D
![Screenshot](figure1.png)
## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

