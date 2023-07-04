import torch
from torch.distributions import Normal
import numpy as np
from scipy.integrate import quad
import os
import pdb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import math
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
import matplotlib.path as mpath
import argparse
def truncated_normal_pdf(x, mean, std, a, b):
    """
    Calculates the probability density function (PDF) of a truncated normal distribution.

    Args:
        x (tensor): The input tensor.
        mean (float): The mean of the underlying normal distribution.
        std (float): The standard deviation of the underlying normal distribution.
        a (float): The lower bound of the truncation range.
        b (float): The upper bound of the truncation range.

    Returns:
        tensor: The PDF values of the truncated normal distribution for the given input tensor.
    """
    # Convert x, mean, std, a, b into tensors
    x = torch.tensor(x)
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    a = torch.tensor(a)
    b = torch.tensor(b)

    z_a = (a - mean) / std
    z_b = (b - mean) / std
    z_x = (x - mean) / std

    if torch.isnan(z_x).any():
        pdb.set_trace() 
    cdf_diff = Normal(0, 1).cdf(z_b) - Normal(0, 1).cdf(z_a)
    truncated_pdf = torch.exp(Normal(0, 1).log_prob(z_x)) / (std * cdf_diff)

    return truncated_pdf


def truncated_normal_cdf(x, mean, std, a, b):
    """
    Calculates the cumulative distribution function (CDF) of a truncated normal distribution.

    Args:
        x (tensor): The input tensor.
        mean (float): The mean of the underlying normal distribution.
        std (float): The standard deviation of the underlying normal distribution.
        a (float): The lower bound of the truncation range.
        b (float): The upper bound of the truncation range.

    Returns:
        tensor: The CDF values of the truncated normal distribution for the given input tensor.
    """
    z_a = (a - mean) / std
    z_b = (b - mean) / std
    z_x = (x - mean) / std

    cdf_diff = Normal(0, 1).cdf(z_b) - Normal(0, 1).cdf(z_a)
    truncated_cdf = (Normal(0, 1).cdf(z_x) - Normal(0, 1).cdf(z_a)) / cdf_diff

    return truncated_cdf

def f_combined(inputs, x):
    result = 0
    prev_term = 1

    for input_set in inputs:
        angle_p, scaler1, mean, std, a, b = input_set
        term = truncated_normal_pdf((x - angle_p) / scaler1, mean, std, a, b)
        result += prev_term * term
        prev_term *= (1 - term)

    return result

def integrate_product_trapezoidal_rule(f, a, b, num_points=1000):
    """
    Numerically integrates the product of a function of multiple variables using the trapezoidal rule.

    Args:
        f (callable): The function to integrate. It should accept a single argument (an array-like object) that represents the variables.
        a (array-like): Lower bounds of the integration interval for each variable.
        b (array-like): Upper bounds of the integration interval for each variable.
        num_points (int): Number of points to sample for the trapezoidal rule.

    Returns:
        float: Approximate integral of the product of the function using the trapezoidal rule.
    """

    num_vars = len(a)
    grid = [np.linspace(a[i], b[i], num_points) for i in range(num_vars)]
    points = np.meshgrid(*grid, indexing='ij')
    values = f(*points)
    integral = np.trapz(values, x=grid[0])
    return integral


def integrate_product_gaussian_quadrature(f, a, b, num_points=100):
    """
    Numerically integrates the product of a function of multiple variables using Gaussian quadrature.

    Args:
        f (callable): The function to integrate. It should accept a single argument (an array-like object) that represents the variables.
        a (array-like): Lower bounds of the integration interval for each variable.
        b (array-like): Upper bounds of the integration interval for each variable.
        num_points (int): Number of points to sample for Gaussian quadrature.

    Returns:
        float: Approximate integral of the product of the function using Gaussian quadrature.
    """
    def product_function(*args):
        return np.prod(f(*args))

    integral, _ = quad(product_function, a, b, points=num_points)
    return integral

def integrate_product_simpsons_rule(f, a, b, num_points=1000):
    """
    Numerically integrates the product of a function of multiple variables using Simpson's rule.

    Args:
        f (callable): The function to integrate. It should accept a single argument (an array-like object) that represents the variables.
        a (array-like): Lower bounds of the integration interval for each variable.
        b (array-like): Upper bounds of the integration interval for each variable.
        num_points (int): Number of points to sample for Simpson's rule.

    Returns:
        float: Approximate integral of the product of the function using Simpson's rule.
    """
    num_vars = len(a)
    grid = [np.linspace(a[i], b[i], num_points) for i in range(num_vars)]
    points = np.meshgrid(*grid, indexing='ij')
    values = f(*points)
    product = np.prod(values, axis=0)

    h = (b - a) / (num_points - 1)
    integral = h[0] * h[1] * np.sum(product[0:num_points-2:2, 0:num_points-2:2] +
                                   4 * product[1:num_points-1:2, 0:num_points-2:2] +
                                   product[2:num_points:2, 0:num_points-2:2] +
                                   4 * product[0:num_points-2:2, 1:num_points-1:2] +
                                   16 * product[1:num_points-1:2, 1:num_points-1:2] +
                                   4 * product[2:num_points:2, 1:num_points-1:2] +
                                   product[0:num_points-2:2, 2:num_points:2] +
                                   4 * product[1:num_points-1:2, 2:num_points:2] +
                                   product[2:num_points:2, 2:num_points:2]) / 9

    return integral

# def is_point_within_area(point, bound_points):
#     x1, y1 = bound_points[0]
#     x2, y2 = bound_points[1]
#     x3, y3 = bound_points[2]
#     x, y = point

#     # Calculate the areas of the triangles formed by the point and the three bound points
#     area_total = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
#     area_1 = abs((x - x1) * (y2 - y1) - (x2 - x1) * (y - y1))
#     area_2 = abs((x - x2) * (y3 - y2) - (x3 - x2) * (y - y2))
#     area_3 = abs((x - x3) * (y1 - y3) - (x1 - x3) * (y - y3))

#     # Check if the sum of the triangle areas is equal to the total area
#     return area_1 + area_2 + area_3 == area_total

def is_point_within_area(point, bound_points):
    x1, y1 = bound_points[0]
    x2, y2 = bound_points[1]
    x3, y3 = bound_points[2]
    x, y = point
    # Define the triangle path
    triangle_path_data = [
        (x1, y1),
        (x2, y2),
        (x3, y3)
    ]
    triangle_path = mpath.Path(triangle_path_data)

    return triangle_path.contains_point((x, y))

def save_triangle_and_point_plot(bound_points, point, filename):
    # Extract the coordinates of the bound points
    x = [p[0] for p in bound_points]
    y = [p[1] for p in bound_points]

    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Plot the point
    ax.plot(point[0], point[1], 'ro', label='Point')

    # Plot the triangle
    ax.plot(x + [x[0]], y + [y[0]], 'b-', label='Triangle')

    # Set the axis limits and labels
    ax.set_xlim(60, 120)
    ax.set_ylim(0, 80)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add legend
    ax.legend()

    # Save the plot to a file
    plt.savefig(filename)

def angle_between_three_points(pointA, pointB, pointC):
    #ref https://muthu.co/using-the-law-of-cosines-and-vector-dot-product-formula-to-find-the-angle-between-three-points/
    BA = pointA - pointB
    BC = pointC - pointB

    try:
        cosine_angle = round(np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC)),8)
        angle = np.arccos(cosine_angle)
    except:
        print("exc")
        raise Exception('invalid cosine')

    return np.degrees(angle)

def distance_between_two_points(pointA, pointB):
    return np.linalg.norm(pointA - pointB)

def point_position(point, line_point1, line_point2):
    vector1 = line_point2 - line_point1
    vector2 = point - line_point1

    cross_product = np.cross(vector1, vector2)

    if cross_product > 0:
        return "Above"
    elif cross_product == 0:
        return "On the line"
    else:
        return "Below"

def calculate_shot_block_prob(index,train,scaler_1=1,scaler_2=1,scaler_3=1,mean=0,sigma=0.4,a=-10,b=10):
    pose_left_x=120
    pose_left_y=36
    pose_right_x=120
    pose_right_y=44

    #get the location of the shooter
    shoter_x=train["location_x"][index]
    shoter_y=train["location_y"][index]
    #get the location of the other players
    other_player={}
    for other_player_num in range(22):
        if not pd.isnull(train["player"+str(other_player_num)+"_location_x"][index]):
            # print(index,other_player,train["player"+str(other_player)+"_location_x"][index])
            other_player[other_player_num]={"x":train["player"+str(other_player_num)+"_location_x"][index],"y":train["player"+str(other_player_num)+"_location_y"][index],
                                        "teammate":train["player"+str(other_player_num)+"_teammate"][index],"position_name":train["player"+str(other_player_num)+"_position_name"][index]}

    #filter non teammate, Goalkeeper, and player not in between the shooter and the goal
    filtered_player = {k: v for k, v in other_player.items() if v.get('position_name') != 'Goalkeeper'}
    filtered_player = {k: v for k, v in filtered_player.items() if v.get('teammate') == False}
    keys_to_remove = []
    for k, v in filtered_player.items():
        filter=is_point_within_area((filtered_player[k]["x"], filtered_player[k]["y"]),[(pose_left_x, 30), (pose_right_x, 50), (shoter_x, shoter_y)])
        if not filter:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del filtered_player[k]

    #272 row with player in between the shooter and the goal, total:2060 row

    # print("index",index,"len(filtered_player)",len(filtered_player),filtered_player)

    #model the shot block probability
    if len(filtered_player)>0:

        #calculate the angle from the shooter to the goal left post and shooter to the player
        pointA = np.array([pose_left_x, pose_left_y])
        pointB = np.array([shoter_x,shoter_y])
        pointC = np.array([pose_right_x, pose_right_y])
        total_angle=angle_between_three_points(pointA, pointB, pointC)
        # print("total_angle :",total_angle)

        #calculate the angle between the shooter to the goal left post and shooter to the player
        for k, v in filtered_player.items():
            pointC = np.array([v['x'],v['y']])
            defender_angle=angle_between_three_points(pointA, pointB, pointC)
            defender_distance=distance_between_two_points(pointB, pointC)
            point_position_result=point_position(pointC, pointB, pointA)
            if point_position_result=="Below":
                defender_angle=-defender_angle #if the player is below the line of the shooter and the left pose in the normal coordination, the angle is negative
            v['defender_angle'] = defender_angle
            v['defender_distance'] = defender_distance
        
        #sort the player by distance to the shooter
        sorted_players = sorted(filtered_player.items(), key=lambda item: item[1]['defender_distance'])
        filtered_player = {k: v for k, v in sorted_players}

        #calculate the probability of the shot block
        list(filtered_player.keys())
        function_list=[]
        for k, v in filtered_player.items(): 
            angle_p=v['defender_angle']
            std=sigma+v['defender_distance']/scaler_2
            function_list.append([angle_p,scaler_1,mean,std,a,b])

        integral = integrate_product_trapezoidal_rule(lambda *args: f_combined(function_list, *args), [0], [total_angle], num_points=1000)
        prediction=(integral/total_angle)*scaler_3 
    else:
        prediction=0
    return prediction

def optimize_function(params):
    prediction_list = []
    for index, row in train.iterrows():
        #calculate the probability of the shot block for each row
        prediction = calculate_shot_block_prob(index, train, scaler_1=params[0], scaler_2=params[1], scaler_3=params[2], mean=0, sigma=params[3], a=params[4], b=-params[4])
        prediction_list.append(prediction)
    #calculate the logloss
    logloss = log_loss(train[result], prediction_list)
    print(logloss)
    return logloss

def print_iteration(iteration):
    print("Iteration:", iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--method", help="method",default="powell")
    args = parser.parse_args()
    print("method:",args.method)
    data=pd.read_csv("/home/c_yeung/workspace6/python/project3/data/train.csv")

    df=data[:]
    #target
    df['shotblock_y'] = df['shot_outcome_grouped'].apply(lambda x: 1 if x == 'Blocked' else 0)
    #drop shot_outcome_grouped
    df=df.drop(["shot_outcome_grouped"],axis=1)
    df['shot_aerial_won'] = df['shot_aerial_won'].apply(lambda x: 1 if x == True else 0)
    df['shot_first_time'] = df['shot_first_time'].apply(lambda x: 1 if x == True else 0)
    df['shot_follows_dribble'] = df['shot_follows_dribble'].apply(lambda x: 1 if x == True else 0)
    df['shot_open_goal'] = df['shot_open_goal'].apply(lambda x: 1 if x == True else 0)
    df['under_pressure'] = df['under_pressure'].apply(lambda x: 1 if x == True else 0)
    df['shot_one_on_one'] = df['shot_one_on_one'].apply(lambda x: 1 if x == True else 0)
    result=["shotblock_y"]


    data_out=pd.DataFrame(columns=["model","loss1","loss2","loss3","loss4","loss5","avg","time"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
    # Define the initial parameter values
    initial_params = [1, 1, 1, 0.4, -10]

    # Perform parameter optimization for each split
    optimized_params_list = []
    logloss_list = []
    valid_logloss_list = []
    for i, (train_index, test_index) in enumerate(skf.split(df.drop('shotblock_y', axis=1), df[result])):

        method = args.method
        print("method:",method)
        train,test=df.iloc[train_index], df.iloc[test_index]
        # Perform parameter optimization
        opt_result = minimize(optimize_function, initial_params, method=method, callback=print_iteration) #options={'maxiter': 1}
        # Obtain the optimized parameters and log loss
        optimized_params = opt_result.x
        logloss = opt_result.fun
        # optimized_params=[20.01726205,  4.77671054,  0.95054114,  0.27260831, -7.73604756]
        # logloss=0.8511282749294645

        # Store the optimized parameters and log loss
        optimized_params_list.append(optimized_params)
        logloss_list.append(logloss)

        print(optimized_params,logloss)

        prediction_list = []
        for index, row in train.iterrows():
            #calculate the probability of the shot block for each row
            prediction = calculate_shot_block_prob(index, train, scaler_1=optimized_params[0], scaler_2=optimized_params[1], scaler_3=optimized_params[2], mean=0, sigma=optimized_params[3], a=optimized_params[4], b=-optimized_params[4])
            prediction_list.append(prediction)
        #calculate the logloss
        logloss = log_loss(train[result], prediction_list)
        print(logloss)
        # #convert the probability to 0 or 1
        # prediction_list_binary=[1 if x>0.5 else 0 for x in prediction_list]
        # cm= confusion_matrix(train[result],prediction_list_binary)
        # class_labels = ['True 0', 'True 1']
        # # Print the confusion matrix with labels
        # print("Confusion Matrix:")
        # for i in range(len(class_labels)):
        #     row_label = class_labels[i] + " Predicted"
        #     row = cm[i, :]
        #     print("{:<15}: {}".format(row_label, row))
        # cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm_percent = np.round(cm_percent, decimals=2)
        # print("confusion matrix :",cm_percent)
    #validation

        valid_predition = []
        for index, row in test.iterrows():
            #calculate the probability of the shot block for each row
            prediction = calculate_shot_block_prob(index, test, scaler_1=optimized_params[0], scaler_2=optimized_params[1], scaler_3=optimized_params[2], mean=0, sigma=optimized_params[3], a=optimized_params[4], b=-optimized_params[4])
            valid_predition.append(prediction)
        #calculate the logloss
        logloss = log_loss(test[result], valid_predition)
        print(logloss)
        valid_logloss_list.append(logloss)

    #save the result to csv with the list
    out_df = pd.DataFrame({'train_loss': logloss_list, 'valid_loss': valid_logloss_list, 'param': optimized_params_list})
    out_df.to_csv(f'/home/c_yeung/workspace6/python/project3/result/shotblock/rule_based/result_{args.method}.csv', index=False)

#[36.94630071 12.3578812   0.4997678   0.15766148 -2.30980703]

    # for i in range(len(optimized_params_list)):
    #     print("Optimized Parameters (Split {}):".format(i), optimized_params_list[i])
    #     print("Log Loss (Split {}):".format(i), logloss_list[i])



        # #convert the probability to 0 or 1
        # prediction_list_binary=[1 if x>0.5 else 0 for x in prediction_list]
        # cm= confusion_matrix(train[result],prediction_list_binary)
        # class_labels = ['True 0', 'True 1']
        # # Print the confusion matrix with labels
        # print("Confusion Matrix:")
        # for i in range(len(class_labels)):
        #     row_label = class_labels[i] + " Predicted"
        #     row = cm[i, :]
        #     print("{:<15}: {}".format(row_label, row))
        # cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm_percent = np.round(cm_percent, decimals=2)
        # print("confusion matrix :",cm_percent)
        # pdb.set_trace()



