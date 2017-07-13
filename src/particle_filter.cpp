/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "particle_filter.h"

using namespace std;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

double sqr(double x) { return pow(x, 2); }

double dSqr(const LandmarkObs& a, const LandmarkObs& b) {
  return sqr(b.x - a.x) + sqr(b.y - a.y);
}

double dSqr(const Particle& a, const LandmarkObs& b) {
  return sqr(b.x - a.x) + sqr(b.y - a.y);
}

double gaussianPDF2(double mu1, double mu2, double std1, double std2, double x1,
                    double x2) {
  return exp(-0.5 * (sqr(x1 - mu1) / std1 + sqr(x2 - mu2) / std2));
}

// I assume std_landmark corresponds to x and y standard deviation as in
// main.cpp, not range and bearing as in particle_filter.h
double gaussianPDF(const vector<LandmarkObs>& predictions,
                   const vector<LandmarkObs>& observations,
                   double std_landmark[]) {
  double density = 1.0;
  // I assume predictions are ordered by id and id goes from 1 to
  // predictions.size()
  for (const LandmarkObs& observation : observations) {
    const LandmarkObs& closestLandmark = predictions[observation.id - 1];
    assert(closestLandmark.id == observation.id);

    density *=
        gaussianPDF2(closestLandmark.x, closestLandmark.y, std_landmark[0],
                     std_landmark[1], observation.x, observation.y);
  }

  density *= pow(kTwoPi * std_landmark[0] * std_landmark[1],
                 -0.5 * observations.size());

  return density;
}

pair<double, double> rotateThenTranslate(double x, double y, double theta,
                                         double xt, double yt) {
  return make_pair(x * cos(theta) - y * sin(theta) + xt,
                   x * sin(theta) + y * cos(theta) + yt);
}

LandmarkObs convertObservationToMapCoordinates(const Particle& particle,
                                               const LandmarkObs& observation) {
  auto xy = rotateThenTranslate(observation.x, observation.y, particle.theta,
                                particle.x, particle.y);
  return {observation.id, xy.first, xy.second};
}

LandmarkObs singleLandmarkToLandmarkObs(
    const Map::single_landmark_s& landmark) {
  return {landmark.id_i, double(landmark.x_f), double(landmark.y_f)};
}

// Better initialize it once not at every function call
default_random_engine random_engine;
}  // namespace

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;

  weights = std::vector<double>(num_particles, 1.0);

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  particles.reserve(num_particles);
  for (int i = 0; i < num_particles; i++) {
    particles.push_back({i, dist_x(random_engine), dist_y(random_engine),
                         dist_theta(random_engine), 1});
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  for (Particle& p : particles) {
    p.x += dist_x(random_engine);
    p.y += dist_x(random_engine);
    p.theta += dist_x(random_engine);

    double new_x, new_y, new_theta;

    if (yaw_rate < 0.001) {
      new_x = p.x + velocity * cos(p.theta) * delta_t;
      new_y = p.y + velocity * sin(p.theta) * delta_t;
      new_theta = p.theta;
    } else {
      new_x = p.x +
              velocity / yaw_rate *
                  (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      new_y = p.y +
              velocity / yaw_rate *
                  (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      new_theta = p.theta + yaw_rate * delta_t;
    }

    p.x = new_x;
    p.y = new_y;
    p.theta = new_theta;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  assert(!predicted.empty());

  for_each(observations.begin(), observations.end(),
           [&predicted](LandmarkObs& observation) {
             observation.id = min_element(predicted.begin(), predicted.end(),
                                          [&observation](const LandmarkObs& a,
                                                         const LandmarkObs& b) {
                                            return dSqr(a, observation) <
                                                   dSqr(b, observation);
                                          })
                                  ->id;
           });
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  assert(!map_landmarks.landmark_list.empty());

  double sqr_range = sqr(sensor_range * 1.1);
  vector<LandmarkObs> map_predictions(map_landmarks.landmark_list.size());
  transform(map_landmarks.landmark_list.begin(),
            map_landmarks.landmark_list.end(), map_predictions.begin(),
            singleLandmarkToLandmarkObs);

  vector<LandmarkObs> map_observations(observations.size());
  int i = 0;
  for (Particle& particle : particles) {
    vector<LandmarkObs> map_predictions_in_range;

    for (const LandmarkObs& map_prediction : map_predictions) {
      if (dSqr(particle, map_prediction) < sqr_range) {
        map_predictions_in_range.push_back(map_prediction);
      }
    }

    if (map_predictions_in_range.empty()) {
      map_predictions_in_range.push_back(map_predictions[0]);
    }

    transform(
        observations.begin(), observations.end(), map_observations.begin(),
        bind(convertObservationToMapCoordinates, particle, placeholders::_1));

    dataAssociation(map_predictions_in_range, map_observations);

    // DEBUG
    vector<int> associations(map_observations.size());
    vector<double> sense_x(map_observations.size());
    vector<double> sense_y(map_observations.size());
    int j = 0;
    for (LandmarkObs& map_observation : map_observations) {
      associations[j] = map_observation.id;
      sense_x[j] = map_observation.x;
      sense_y[j] = map_observation.y;
      j++;
    }
    particle = SetAssociations(particle, associations, sense_x, sense_y);

    particle.weight =
        gaussianPDF(map_predictions, map_observations, std_landmark);
    weights[i] = particle.weight;
    i++;
  };
}

void ParticleFilter::resample() {
  discrete_distribution<> dist(weights.begin(), weights.end());

  vector<Particle> new_particles(particles.size());
  generate(new_particles.begin(), new_particles.end(),
           [&]() { return particles[dist(random_engine)]; });

  particles.swap(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
