//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019
//

#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

// isce3::core
#include <isce3/core/Constants.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/LUT1d.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/StateVector.h>

// isce3::geometry
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/geometry.h>

// flatten namespaces
using isce3::core::LookSide;
const auto Left = LookSide::Left;
const auto Right = LookSide::Right;

// convenience enum to convey semantics of sign on spin rate
enum OrbitDirection { EastBound, WestBound };
enum OrbitDirection orbDir(double omega) {
    if (omega > 0) return EastBound;
    return WestBound;
}

struct GeometryTest : public ::testing::Test {

    // isce3::core objects
    isce3::core::Ellipsoid ellipsoid;
    isce3::core::Orbit orbit;
    double hsat;
    double satlat0;
    double satomega;


    // Constructor
    GeometryTest(){}

    // Setup an orbit moving eastbound in a circle about the Z-axis at constant
    // angular rate omega = dlon/dt.
    void Setup_data(double lat0, double lon0, double omega, int Nvec)
    {
        //WGS84 ellipsoid
        ellipsoid = isce3::core::Ellipsoid(6378137.,.0066943799901);

        //Satellite height
        hsat = 700000.0;

        //Save constants
        satlat0 = lat0;
        satomega = omega;

        //Setup orbit
        isce3::core::DateTime t0("2017-02-12T01:12:30.0");
        orbit.referenceEpoch(t0);

        double clat = std::cos(lat0);
        double slat = std::sin(lat0);
        double sath = ellipsoid.a() + hsat;

        std::vector<isce3::core::StateVector> statevecs(Nvec);
        for(int ii=0; ii < Nvec; ii++)
        {
            double deltat = ii * 10.0;
            double lon = lon0 + omega * deltat;
            isce3::core::Vec3 pos, vel;

            pos[0] = sath * clat * std::cos(lon);
            pos[1] = sath * clat * std::sin(lon);
            pos[2] = sath * slat;

            vel[0] = -omega * pos[1];
            vel[1] = omega * pos[0];
            vel[2] = 0.0;

            isce3::core::DateTime epoch = t0 + deltat;

            statevecs[ii].datetime = epoch;
            statevecs[ii].position = pos;
            statevecs[ii].velocity = vel;
        }
        orbit.setStateVectors(statevecs);
    }

    //Solve for Geocentric latitude given a slant range and look side.
    double solve(double R, LookSide side)
    {
        double temp = 1.0 + hsat/ellipsoid.a();
        double temp1 = R/ellipsoid.a();
        double temp2 = R/(ellipsoid.a() + hsat);

        double cosang = 0.5 * (temp + (1.0/temp) - temp1 * temp2);
        double angdiff = std::acos(cosang);

        double x;
        if ( ((Left == side) && (EastBound == orbDir(satomega)))
                || ((Right == side) && (WestBound == orbDir(satomega))) ) {
            x = satlat0 + angdiff;
        } else {
            x = satlat0 - angdiff;
        }
        return x;

    }
};

TEST_F(GeometryTest, RdrToGeoLat) {

    // Loop over test data
    const double degrees = 180.0 / M_PI;

    //Moving at 0.1 degrees / sec
    const double lon0 = 0.0;
    const double omega = 0.1/degrees;
    const int Nvec = 10;
    const double lat0 = 45.0/degrees;
    LookSide sides[] = {Right, Left};

    //Set up orbit
    Setup_data(lat0, lon0, omega, Nvec);

    //Test over 20 points
    for (size_t ii = 0; ii < 20; ++ii)
    {
        //Azimuth time
        double tinp = 5.0 + ii * 2.0;

        //Slant range
        double rng = 800000. + 10.0 * ii;

        //Theoretical solutions
        double expectedLon = lon0 + omega * tinp;


        for(int kk=0; kk<2;kk++)
        {
            //Expected solution for geocentric latitude
            double geocentricLat = solve(rng, sides[kk]);

            //Convert geocentric coords to xyz
            isce3::core::cartesian_t xyz = { ellipsoid.a() * std::cos(geocentricLat) * std::cos(expectedLon),
                                            ellipsoid.a() * std::cos(geocentricLat) * std::sin(expectedLon),
                                            ellipsoid.a() * std::sin(geocentricLat)};

            //Convert xyz to geodetic coords
            isce3::core::cartesian_t expLLH;
            ellipsoid.xyzToLonLat(xyz, expLLH);

            //Set up DEM to match expected height
            isce3::geometry::DEMInterpolator dem(expLLH[2]);

            // Initialize guess
            isce3::core::cartesian_t targetLLH = {0.0, 0.0, 0.0};

            // Run rdr2geo
            int stat = isce3::geometry::rdr2geo(tinp, rng, 0.0,
                        orbit, ellipsoid, dem, targetLLH, 0.24, sides[kk],
                        1.0e-8, 25, 15);

            // Check
            ASSERT_EQ(stat, 1);
            ASSERT_NEAR(targetLLH[0], expLLH[0], 1.0e-8);
            ASSERT_NEAR(targetLLH[1], expLLH[1], 1.0e-8);
            ASSERT_NEAR(targetLLH[2], expLLH[2], 1.0e-3);
        }
    }

}


TEST_F(GeometryTest, GeoToRdrLat) {


    // Loop over test data
    const double degrees = 180.0 / M_PI;

    //Moving at 0.1 degrees / sec
    const double lon0 = 0.0;
    const double omega = 0.1/degrees;
    const int Nvec = 10;
    const double lat0 = 45.0/degrees;
    LookSide sides[] = {Right, Left};

    //Set up orbit
    Setup_data(lat0, lon0, omega, Nvec);

    //Constant zero Doppler
    isce3::core::LUT2d<double> zeroDoppler;

    // Dummy wavelength
    const double wavelength = 0.24;

    //Test over 19 points
    for (size_t ii = 1; ii < 20; ++ii)
    {
        //Azimuth time
        double tinp = 25.0 + ii * 2.0;


        for (int kk=0; kk<2; kk++)
        {
            //Start with geocentric lat
            double geocentricLat = 0;
            if ( ((Left == sides[kk]) && (EastBound == orbDir(satomega)))
                    || ((Right == sides[kk]) && (WestBound == orbDir(satomega))) ) {
                geocentricLat = (lat0 + ii * 0.1/degrees) ;
            } else {
                geocentricLat = (lat0 - ii * 0.1/degrees) ;
            }

            //Theoretical solutions
            double expectedLon = lon0 + omega * tinp;

            //Convert geocentric coords to xyz
            isce3::core::cartesian_t targ_xyz = { ellipsoid.a() * std::cos(geocentricLat) * std::cos(expectedLon),
                                                 ellipsoid.a() * std::cos(geocentricLat) * std::sin(expectedLon),
                                                 ellipsoid.a() * std::sin(geocentricLat)};

            //Transform to geodetic LLH
            isce3::core::cartesian_t targ_LLH;
            ellipsoid.xyzToLonLat(targ_xyz, targ_LLH);

            //Expected satellite position
            isce3::core::cartesian_t sat_xyz = { (ellipsoid.a() + hsat) * std::cos(expectedLon) * std::cos(lat0),
                                                (ellipsoid.a() + hsat) * std::sin(expectedLon) * std::cos(lat0),
                                                (ellipsoid.a() + hsat) * std::sin(lat0)};

            const isce3::core::cartesian_t los = sat_xyz - targ_xyz;

            //Expected slant range
            double expRange = los.norm();

            // Run geo2rdr
            double aztime, slantRange;
            int stat = isce3::geometry::geo2rdr(targ_LLH, ellipsoid, orbit,
                zeroDoppler, aztime, slantRange, wavelength, sides[kk],
                1.0e-9, 50, 10.0);

           // Check
            ASSERT_EQ(stat, 1);
            ASSERT_NEAR(aztime, tinp, 1.0e-5);
            ASSERT_NEAR(slantRange, expRange, 1.0e-8);
        }
    }
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


/*

# Description
-------------

This unit test compares output of geometry algorithms against analytic solutions
derived for a satellite flying at constant velocity and radius over a line of constant
latitude.

The target is assumed to lie on a sphere with radius equal to Ellipsoid's major axis.


## Geodetic LLH to ECEF XYZ
---------

Radius along the East-West direction ($R_e$) is given by:

$$R_e \left(\theta \right) = \frac{a}{\sqrt{1 - e^2 \cdot \sin^2 \left(\theta \right)}}$$


Using the East-West radius, a given target at Geodetic Latitude ($\theta$), Longitude ($\psi$)
and Height ($h$) can be transformed to Caresian ECEF coordinates as follows:

$$X = \left( R_e\left( \theta \right) + h \right) \cdot \cos \theta \cdot \cos \psi$$
$$Y = \left( R_e\left( \theta \right) + h \right) \cdot \cos \theta \cdot \sin \psi$$
$$Z = \left( R_e\left( \theta \right) \cdot \left( 1 - e^2 \right) +h \right) \cdot \sin \theta $$


## Parametric form of Sphere with Geocentric Latitude
-----------

A point $\vec{S}$ on a sphere at a height $h$ above the major axis characterized by Geocentric
Latitude ($\lambda$), Longitude ($\psi$) can be expressed in ECEF coordinates as follows:

$$ X = \left( a + h \right) \cdot \cos \lambda \cos \psi $$
$$ Y = \left( a + h \right) \cdot \cos \lambda \sin \psi $$
$$ Z = \left( a + h \right) \cdot \sin \lambda $$


## Target on same longitude as satellite is on Zero Doppler Contour
----------

Consider a target ($\vec{T}$) located on the same longitude as the satellite.
Let the location of target ($\vec{T}$) be represented by geocentric latitude $\lambda$,
longitude $\psi$ and zero height .

$$X_t = a \cdot \cos \lambda \cdot \cos \psi$$
$$Y_t = a \cdot \cos \lambda \cdot \sin \psi$$
$$Z_t = a \cdot \sin \lambda $$


Using the above expressions, it can be shown that

$$\left( \vec{R_{s}} - \vec{T} \right) \cdot \vec{V_{s}} = 0$$

Hence, it is sufficient to solve for Target latitude ($\lambda$) when estimating target
on reference surface of ellipsoid ($h_t$) for a given slant range for forward
geometry operations.


## Target on Geocentric sphere with radius a
-----------

For a given slant range ($R$), we can write out

$$ \left( \left( a + h_s \right) \cdot \cos \lambda_s  -   a \cdot \cos \lambda \right)^2$$
$$ + \left( \left( a +h_s \right) \cdot \sin \lambda_s - a \cdot \sin \lambda \right)^2 = R^2$$

Leading to

$$\cos \left( \lambda - \lambda_s \right) = \frac{1}{2} \cdot \left[ \frac{a+h_s}{a} + \frac{a}{a+h_s} - \frac{R}{a} \cdot \frac{R}{a+h_s} \right] $$

*/



// end of file
