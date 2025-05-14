# Computes errorbars for spectra with point-to-point scattering using quadratic approximation
# Created by Alexandrine L'Heureux and modified by Mathis Bouffard

# ATTENTION: this code outputs a p-t-p scatter that is too high when the density of points is very low
# This should not be a problem for JWST or high resolution observations

# This program has been tested and works for:
    # different input functions as y
    # different levels of noise added to y
    # different amounts of data points
    # different domain sizes


import numpy as np



def diff_quad(x, y):
    '''
    Calculates the difference between each data point in a spectrum and the expected value from a quadratic fit.
    x: wavelength
    y: flux
    '''
    
    fitted_x = []
    difference_quad = []
    
    for i in range(len(x))[2:-1]:   # some points must be discarded because the quadratic approximation can't reach
                                    # the limit of the domain
        fitted_x.append(x[i])

        # Selecting points to fit order-4 polynomial
        x_fit = np.copy(x[i - 2:i + 2])
        x_fit -= x[i]  # bring point of interest (x3) to 0
        assert len(x_fit) == 4
        x1, x2, x3, x4 = x_fit[0], x_fit[1], x_fit[2], x_fit[3]

        y_fit = y[i - 2:i + 2]
        y1, y2, y3, y4 = y_fit[0], y_fit[1], y_fit[2], y_fit[3]
        assert len(y_fit) == 4

        # Calculate the expected value of y3
        c = y1 - x1 * (y2 - y4) / (x2 - x4) - (x1 ** 2 + (x1 * (x4 ** 2 - x2 ** 2) / (x2 - x4))) * (
                y4 - y1 + (y2 - y4) * (x1 - x4) / (x2 - x4)) * (
                    x4 ** 2 - x1 ** 2 + (x4 ** 2 - x2 ** 2) * (x4 - x1) / (x2 - x4)) ** (-1)
        difference_quad.append(y3 - c)
        
    return np.array(fitted_x), np.array(difference_quad)

 


def ptp_scatter(x, y, dom_size=20):
    '''
    rewrite this docstring using the following info
    
    # Computes the point-to-point scatter by splitting the spectrum into many smaller domains.
    # The uncertainty found for the point at the middle of each
    # domain will be used as the error for all the points inside the domain
    
    # Computes the y-uncertainty for the point in the middle of an array[start_domain:end_domain] using quadratic approximation
    # start_domain and end_domain must be index values of x and y
    # For spectra, x is wavelength and y is flux
    '''
    
    # Calculate the difference between each data point and the quadratic fit
    fitted_x, diff = diff_quad(x, y)
    
    
    # if start_domain is None: start_domain = 0
    # if end_domain is None: end_domain = len(x) - 1   
    
    # Create arrays of indices for the start and end of every domain of size dom_size
    indices = np.arange(len(x))
    start_domain = np.arange(indices[0], indices[-dom_size], dom_size)  # indices of the start of each domain
    end_domain = np.arange(indices[dom_size], indices[-1], dom_size)  # indices of the end of each domain
    
    if start_domain[-1] == end_domain[-1]:  # if end_domain is shorter than start_domain by 1 value
        end_domain = np.append(end_domain, end_domain[-1] + dom_size)
    
    end_domain[-1] = indices[-1]

    
    # Compute errorbars
    sigmas = np.zeros(len(fitted_x))  # array to store errorbar values for the flux

    
    # Loop over each domain
    for i in range(len(start_domain)):
        
        # Consider only the domain of interest
        x_dom = fitted_x[start_domain[i]:end_domain[i]]
        diff_dom = diff[start_domain[i]:end_domain[i]]

        # Clip outliers > 3 sigmas within this domain (assumes the domain is large enough for that)
        while np.sum(np.abs(diff_dom) > 3 * np.nanstd(diff_dom)) > 0:
            mask_outliers = np.abs(diff_dom) <= 3 * np.nanstd(diff_dom)
            diff_dom = diff_dom[mask_outliers]
            x_dom = x_dom[mask_outliers]
        # print('Outliers at difference > 3 sigmas have been removed')
        
        
        # To avoid division by zero in the rms difference calculation
        if np.count_nonzero(~np.isnan(diff_dom)) == 0:  # if there are only nans in the domain
            # Set the sigmas for this domain as nan and skip to the next domain
            sigmas[start_domain[i]:end_domain[i]] = np.full(sigmas[start_domain[i]:end_domain[i]].size, np.nan)
            continue
        
        
        # Calculate rms for this domain
        rms_difference = np.sqrt((1 / np.count_nonzero(~np.isnan(diff_dom))) * np.nansum(diff_dom ** 2))
        
        
        # Select values at the center of the domain
        center_idx = (end_domain[i] - start_domain[i]) // 2
        X1, X2, X3, X4 = np.copy(fitted_x[center_idx - 2:center_idx + 2] - fitted_x[center_idx])

        # Use the values of the 4 points in the middle of the domain to calculate uncertainty
        K = (X1 ** 2 + X1 * (X4 ** 2 - X2 ** 2) / (X2 - X4)) * (
                X4 ** 2 - X1 ** 2 + (X4 ** 2 - X2 ** 2) * (X4 - X1) / (X2 - X4)) ** (-1)
        delC_y1 = 1 + K
        delC_y2 = -X1 / (X2 - X4) - K * (X1 - X4) / (X2 - X4)
        delC_y4 = X1 / (X2 - X4) - K + K * (X1 - X4) / (X2 - X4)
        sigma = rms_difference * np.sqrt(1 + delC_y1 ** 2 + delC_y2 ** 2 + delC_y4 ** 2)**(-1)  # Uncertainty
        
        # Store the errorbars inside the arrays (assuming that sigma is the same for every point within the domain)
        sigmas[start_domain[i]:end_domain[i]] = np.repeat(sigma, sigmas[start_domain[i]:end_domain[i]].size)
        
    
    return sigmas