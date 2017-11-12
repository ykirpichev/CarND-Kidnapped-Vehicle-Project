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

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles.
    num_particles = 100;
	// Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	weights = std::vector<double>(num_particles, 1);
	particles = std::vector<Particle>(num_particles);

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}



void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.

	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen;

	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);
    for (int i = 0; i < num_particles; ++i) {
		auto& p = particles[i];
		p.x = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen) * delta_t;
		p.y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + dist_y(gen) * delta_t;
		p.theta = p.theta + yaw_rate * delta_t + dist_theta(gen) * delta_t;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	for (auto&& observation: observations) {
		observation.id = 0;
		auto bestDist = dist(predicted[0].x, predicted[0].y, observation.x, observation.y);
		for (int i = 0; i < predicted.size(); ++i) {
			auto currDist = dist(predicted[i].x, predicted[i].y, observation.x, observation.y);
			if (currDist < bestDist) {
				bestDist = currDist;
				observation.id = i;
			}
		}
	}
}


double gaussian(const LandmarkObs& obs, const LandmarkObs& pred, double std_land[]) {
    double x = obs.x;
	double y = obs.y;
	double mx = pred.x;
	double my = pred.y;
	double sigmax = std_land[0];
	double sigmay = std_land[1];
	double exponenta = pow(x - mx, 2) / (2 * sigmax * sigmax) + pow(y - my, 2) / (2 * sigmay * sigmay);
    return 1 / (2 * M_PI * sigmax * sigmay) * exp(-exponenta);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

	for (int i = 0; i < num_particles; ++i) {
		auto&& particle = particles[i];
		std::vector<LandmarkObs> predicted;
		for (auto&& mark : map_landmarks.landmark_list) {
			if (fabs(mark.x_f - particle.x) < sensor_range &&
				fabs(mark.y_f - particle.y) < sensor_range) {
				predicted.push_back({mark.id_i, mark.x_f, mark.y_f});
			}
		}

		std::vector<LandmarkObs> observed;
		// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
		//   according to the MAP'S coordinate system.
		//   Need to transform between the two systems.
		//   The following is a good resource for the theory:
		//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
		//   http://planning.cs.uiuc.edu/node99.html
		for (auto&& obs: observations) {
			double mapX = particle.x + cos(particle.theta) * obs.x - sin(particle.theta) * obs.y;
			double mapY = particle.y + sin(particle.theta) * obs.x + cos(particle.theta) * obs.y;
			observed.push_back({0, mapX, mapY});
		}

		dataAssociation(predicted, observed);

		// update weight using multivariate normal distribution
		double weight = 1.0;
		for (auto&& obs: observed) {
			weight *= gaussian(obs, predicted[obs.id], std_landmark);
		}
		weights[i] = weight;
	}
}

void ParticleFilter::resample() {
	/*
	 * Implementation from lections
	 * p3 = []
	 * index = int(random.random() * N)
	 * beta = 0
	 * mw2 = 2 * max(w)
	 * for i in range(N):
	 *     beta += random.random() * mw2
	 *     while w[index] < beta:
	 *         beta -= w[index]
	 *         index = (index + 1) % N
	 *
	 *     p3.append(p[index])
	 *
	 * p = p3
	 */

	// Try to use std::discrete_distribution
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine gen;
	std::discrete_distribution<> d(weights.begin(), weights.end());
    std::vector<Particle> newParticles;
	for (int i = 0; i < num_particles; ++i) {
		auto index = d(gen);
		newParticles.push_back(particles[index]);
	}
	particles.swap(newParticles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
