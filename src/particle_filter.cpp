/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define REALLY_BIG_DISTANCE 9999999

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;

	//Set standard deviations for x, y, and theta.
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];


	// Create a normal (Gaussian) distributions.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);



	for (int i = 0; i < this->num_particles; ++i) {

		double sample_x, sample_y, sample_theta;

		// Sample  and from these normal distrubtions like this: 
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);


		// Create a new particle
		Particle newParticle = Particle();
		newParticle.id = i;
		newParticle.x = sample_x;
		newParticle.y = sample_y;
		newParticle.theta = sample_theta;
		newParticle.weight = 1;

		//Add the new particle to the particles vector
		particles.push_back(newParticle);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	//Set standard deviations for x, y, and theta.
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];


	// Create a normal (Gaussian) distributions.
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y );
	normal_distribution<double> dist_theta(0, std_theta);



	for (int i = 0; i < this->num_particles; ++i) {

		

		Particle particle = particles[i];

		//pull out terms for readability
		double px = particle.x;
		double py = particle.y;
		double v = velocity;
		double yaw = particle.theta;
		double yaw_d = yaw_rate;


		double px_new = 0;
		double py_new = 0;
		double yaw_new = 0;


		//update position
		
		if (fabs(yaw_d) < 0.001)//avoid division by zero
		{
			px_new = px + v * cos(yaw)*delta_t;
			py_new = py + v * sin(yaw)*delta_t;
		}
		else
		{
			px_new = px + v / yaw_d * (sin(yaw + yaw_d * delta_t) - sin(yaw));
			py_new = py + v / yaw_d * (-cos(yaw + yaw_d * delta_t) + cos(yaw));
		}

		yaw_new = yaw + yaw_rate * delta_t;

		// generate noise terms for state
		double noise_x = dist_x(gen);
		double noise_y = dist_y(gen);
		double noise_theta = dist_theta(gen);

		//add noise
		px_new += noise_x;
		py_new += noise_y;
		yaw_new += noise_theta;

		particle.x = px_new;
		particle.y = py_new;
		particle.theta = yaw_new;

		particles[i] = particle;

	}


}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> predicted, std::vector<LandmarkObs> observations, Particle &particle) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//I'm not sure what predicted measurement means in this context. So I'm interpreting it my own way
	//predicted = the landmarks within range of the particle (in map coordinates)
	//observations = the observatios transformed into map coordinates


	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;

	//loop through all observations
	for (int i = 0; i < observations.size(); i++)
	{
		//reset variables
		int closestIndex = -1;
		double bestDistance = REALLY_BIG_DISTANCE;
		double bestX = 0;
		double bestY = 0;


		
		//loop through all in range landmarks
		for (int j = 0; j < predicted.size(); j++)
		{
			double currentDistance = dist(observations[i].x, observations[i].y, predicted[j].x_f, predicted[j].y_f);

			if (currentDistance < bestDistance)
			{
				closestIndex = predicted[j].id_i;
				bestDistance = currentDistance;
				bestX = predicted[j].x_f;
				bestY = predicted[j].y_f;
				
			}

		}

		sense_x.push_back(bestX);
		sense_y.push_back(bestY);
		associations.push_back(closestIndex);

	}

	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
	particle.associations = associations;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < this->num_particles; ++i)
	{

		Particle particle = this->particles[i];
		//Convert to map coordinates
		std::vector<LandmarkObs> observations_map = ConvertToMapCooridinates(observations, particle);
		//Associate nearest neighbors
		//std::vector<Map::single_landmark_s> inRangeLandmarks = FindInRangeLandmarks(sensor_range, map_landmarks.landmark_list, particle);
		dataAssociation(map_landmarks.landmark_list, observations_map, particle);
		//Assign new weights
		CalculateUnNormalizedWeights(particle, observations_map, std_landmark);

		particles[i] = particle;
	}
	//normalize weights
	NormalizeWeights();
}


void ParticleFilter::NormalizeWeights()
{
	double sumWeights = 0;
	for (int i = 0; i < this->num_particles; ++i)
	{
		sumWeights += particles[i].weight;
	}

	for (int i = 0; i < this->num_particles; ++i)
	{
		double normalizedWeight = particles[i].weight / sumWeights;
		particles[i].weight = normalizedWeight;
	}
}

void ParticleFilter::CalculateUnNormalizedWeights(Particle &particle, std::vector<LandmarkObs> observations, double std_landmark[])
{
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];


	double weight = 1;

	for (int i = 0; i < observations.size(); i++)
	{

		double mu_x = particle.sense_x[i];
		double mu_y = particle.sense_y[i];
		double x_obs = observations[i].x;
		double y_obs = observations[i].y;

		//calculate normalization term
		double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));

		//calculate exponent
		double exponent = (pow((x_obs - mu_x), 2)) / (2 * pow(sig_x, 2)) + (pow((y_obs - mu_y), 2)) / (2 * pow(sig_y, 2));

		//calculate weight using normalization terms and exponent
		weight *= gauss_norm * exp(-exponent);
	}

	particle.weight = weight;

}

std::vector<Map::single_landmark_s> ParticleFilter::FindInRangeLandmarks(double sensor_range, std::vector<Map::single_landmark_s>  landmarkList, Particle particle)
{
	std::vector<Map::single_landmark_s> inRangeLandmarks;

	for (int i = 0; i < landmarkList.size(); i++)
	{
		double distance = dist(particle.x, particle.y, landmarkList[i].x_f, landmarkList[i].y_f);

		if (distance <= sensor_range)
		{
			inRangeLandmarks.push_back(landmarkList[i]);
		}
	}

	return inRangeLandmarks;
}

std::vector<LandmarkObs> ParticleFilter::ConvertToMapCooridinates(std::vector<LandmarkObs> vehicleCooridinates, Particle particle)
{
	std::vector<LandmarkObs> mapCooridinates;

	for (int i = 0; i < vehicleCooridinates.size(); i++)
	{
		LandmarkObs obsVehicle = vehicleCooridinates[i];

		//car observation coordinates
		float xc = obsVehicle.x;
		float yc = obsVehicle.y;
		//particle position
		float xp = particle.x;
		float yp = particle.y;
		float thetap = particle.theta;

		//calculate map observation coordinates
		float xm = xp + cos(thetap)*xc - sin(thetap)*yc;
		float ym = yp + sin(thetap)*xc + cos(thetap)*yc;

		LandmarkObs newObs = LandmarkObs();

		newObs.x = xm;
		newObs.y = ym;
		newObs.id = obsVehicle.id;

		mapCooridinates.push_back(newObs);
	}

	return mapCooridinates;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	

	//Create list of weights
	std::vector<double> weightsList;

	for (int i = 0; i < this->num_particles; ++i) 
	{
		weightsList.push_back(particles[i].weight);
	}


	
	//Help from
	//https://stackoverflow.com/questions/31153610/setting-up-a-discrete-distribution-in-c
	
	// Setup the random bits
	std::random_device rd;
	std::mt19937 gen(rd());


	// Create the distribution with those weights
	std::discrete_distribution<> d(weightsList.begin(), weightsList.end());
	
	std::vector<Particle> newParticles;


	for (int i = 0; i < this->num_particles; ++i) {

		double sample_x, sample_y, sample_theta;

		Particle newParticle = particles[d(gen)];

		newParticles.push_back(newParticle);

	}

	particles = newParticles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
