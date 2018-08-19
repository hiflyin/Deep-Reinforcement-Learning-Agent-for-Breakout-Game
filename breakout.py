# TODO: 1. set random seed; 2. add bias to weights 3. get rid of constants 4. add main function 5. try replacing relu
# TODO: save game after n batches -say 100 games
import numpy as np
import cPickle as pickle
import gym
import struct
import time

def extract_features(frame):
    # turn 3D RGB model into -> 2D gray array - colours weights formula taken from wikipedia
    gray_frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
    gray_frame = gray_frame[93:194, 8:152] #crop to remove top score bar and bottom bar and unnecessary right/left margins
    gray_frame = gray_frame[::2, ::2]  # skip every other pixel from both dimensions to reduce size
    gray_frame[gray_frame != 0] = 1 # set everything else from background to 1
    return gray_frame.astype(np.float).reshape((1, gray_frame.size)) # turning it into one line


######### simple generic neural network model - will be used for predicting next best gama action given game state ####
# 1. define a hypothesis model for the best action given the change in frame
D = 51 * 72  # input dimensionality: 80x80 grid
H = 150  # number of hidden layer neurons

def define_action_value_model():
    return { "weights": {"input_to_hidden":  np.random.standard_normal(size = (H, D)) / np.sqrt(D), "hidden_to_output":  np.random.standard_normal(size = (1,H)) / np.sqrt(H)},
             "learning_rate": 1e-4,
             "decay_rate": 0.99,
             "reward_discount": 0.99,
             "historic_avg_reward": None,
             "rmsprop_grad": {"input_to_hidden": np.zeros((H, D)), "hidden_to_output": np.zeros((1,H)) }}

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def rectifier_linear_unit(x):
    x[x<0] = 0
    return x

def rectifier_linear_unit_derivative(x):
    x[x<=0] = 0
    return x

def feed_forward_sample(inputs, weights):
    hidden_layer_output = rectifier_linear_unit(np.dot(inputs, weights["input_to_hidden"].T))
    return {"hidden_layer": hidden_layer_output,
            "prediction_action_2": sigmoid(np.dot(hidden_layer_output, weights["hidden_to_output"].T))}

def back_prop_sample(inputs, outputs, error, weights):
    error_output = error
    output_derivative = 1
    total_output_delta = error_output * output_derivative
    error_hidden = np.dot(total_output_delta, weights["hidden_to_output"])
    hidden_derivative = rectifier_linear_unit_derivative(outputs["hidden_layer"])
    total_hidden_delta = error_hidden * hidden_derivative
    #print error_output
    #print total_output_delta.sum() == 0
    #print outputs["hidden_layer"].sum() == 0
    #print total_hidden_delta.sum() == 0
    #print inputs.sum() == 0
    #print "-"*30
    return {"hidden_to_output": np.dot(total_output_delta, outputs["hidden_layer"]),
            "input_to_hidden": np.dot(total_hidden_delta.T, inputs)}

def back_prop_sequence(inputs_seq, outputs_seq, errors_seq, weights):
    summed_deltas = {}
    for k, v in weights.iteritems():
        summed_deltas[k] = np.zeros_like(v)
    for i in range(len(errors_seq)):
        deltas = back_prop_sample(inputs_seq[i], outputs_seq[i], errors_seq[i], weights)
        for k, v in summed_deltas.iteritems():
            summed_deltas[k] += deltas[k]
    return summed_deltas

################ reinforcement learning system components ######################################################
# actions = [1, 2]
prob_thr_choosing_random_action = .01
gamma = 0.99  # discount factor for reward
reward_threshold_alert = 0

def decide_next_action(curr_state_change, weights):
    # see what the model says and then decide if to go random or not
    action_value_model_predictions = feed_forward_sample(curr_state_change, weights)
    action = 2 if np.random.uniform() < action_value_model_predictions["prediction_action_2"][0] else 3
    return action, action_value_model_predictions

# given a state and an action, # return the next state I get in by taking this action and the reward
def take_next_action(action, game):
    game.step(1)
    return game.step(action)

def play_until_game_over(game, model, render= False):
    game_actions_taken = []
    resulted_game_states = [extract_features(game.reset())]
    resulted_game_states_diff = [np.zeros_like(resulted_game_states[-1])]
    resulted_game_rewards = []
    action_value_model_predictions = []
    game_is_over = False
    while(game_is_over == False):
        if render:
            game.render()
            time.sleep(.05)
        action, action_value_model_prediction = decide_next_action(resulted_game_states_diff[-1], model["weights"])
        new_game_state, action_reward, game_is_over, other_info =  take_next_action(action, game)

        game_actions_taken.append(action)
        resulted_game_rewards.append(action_reward)
        resulted_game_states.append(extract_features(new_game_state))
        resulted_game_states_diff.append(resulted_game_states[-1] - resulted_game_states[-2])
        action_value_model_predictions.append(action_value_model_prediction)
    print "total resulted game rewards: {}".format(sum(resulted_game_rewards))
    model["historic_avg_reward"] = sum(resulted_game_rewards) if model["historic_avg_reward"] is None else \
        sum(resulted_game_rewards) * 0.01 + model["historic_avg_reward"] * .99

    return game_actions_taken, resulted_game_states_diff, resulted_game_rewards, action_value_model_predictions

def compute_model_updates_from_game_mistakes( weights, actions_taken, resulted_states_diff, resulted_rewards, model_predictions):
    resulted_rewards = attribute_rewards_to_earlier_actions(resulted_rewards)
    # define model targets to correspond to the actions taken
    targets = np.array([1  if x == 2 else 0 for x in actions_taken ])
    # compute model errors as differences between targets and actions's probabilities
    errors = targets - np.array([x["prediction_action_2"][0][0] for x in model_predictions]).flatten()
    # but what we want is to pick the action with best reward
    # if action had positive reward - the larger the reward - the more we want to increase the error accordingly
    # so that we make the model be more sure about taking that action next time and vice-versa for small positive
    # reward - we may want to magnify the error less because maybe other action could have had larger reward
    # so let the model not be so sure..
    # if action had negative reward then we provide a negative error which tells the model the oposite action was better
    # all these are achieved in short by weighting the errors by the rewards:
    errors *= resulted_rewards
    return back_prop_sequence(resulted_states_diff, model_predictions, errors, weights)

def play_several_games(n_games, game, model, render = False):
    # sum up all updates over several games
    all_games_updates = {}
    for k, v in model["weights"].iteritems():
        all_games_updates[k] = np.zeros_like(v)
    for i in range(n_games):
        print "*"*10 + "  Playing game number {}  ".format(i+1) + "*"*10
        model_updates = compute_model_updates_from_game_mistakes(model["weights"], *play_until_game_over(game, model, render))
        render = False
        for k, v in model_updates.iteritems():
            all_games_updates[k] += v
    print "-" * 10 + " Averaged reward since beggining to learn is {}  ".format(model["historic_avg_reward"]) + "-" * 10
    return all_games_updates

def attribute_rewards_to_earlier_actions(rewards):
    for t in range(1,len(rewards)):
        # in some games - when reward is not 0 - it means afterwards it started from scratch
        # so no future rewards from here should be propagated as there's no dependency between actions from here
        # so only when the reward at time t-1 is 0 we attribute some portion gamma of the reward from future to it
        if rewards[-t-1] == 0: rewards[-t-1] = gamma*rewards[-t]
    # finally we standardize the rewards to a common scale
    rewards -= np.mean(rewards)
    if np.std(rewards) != 0:
        rewards /= np.std(rewards)
    return rewards

def learn_from_mistakes(action_value_model, model_upates):
    for k, v in action_value_model["weights"].iteritems():
        action_value_model["rmsprop_grad"][k] = action_value_model["decay_rate"] * action_value_model["rmsprop_grad"][k] + \
                                                (1 - action_value_model["decay_rate"]) * (model_upates[k] ** 2)
        total = action_value_model["learning_rate"] * model_upates[k] / (np.sqrt(action_value_model["rmsprop_grad"][k]) + 1e-5)
        #print total.sum()
        action_value_model["weights"][k] += total
    return action_value_model

def learn_to_play(games_to_play_before_learning_better_actions = 10):
    game = gym.make("Breakout-v0")
    model_to_predict_game_action = define_action_value_model()

    batch_no = 1
    render = False
    while(True):
        if batch_no %10 == 0: render = True
        print "Playing batch no {}. Total games played is: {}. ".format(batch_no,
                                                        (batch_no-1)*games_to_play_before_learning_better_actions)
        needed_action_value_model_updates = play_several_games(games_to_play_before_learning_better_actions, game,
                                                               model_to_predict_game_action, render)
        learn_from_mistakes(model_to_predict_game_action, needed_action_value_model_updates)
        batch_no += 1
        render = False

learn_to_play(10)