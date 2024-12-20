import numpy as np
import socket
import json
import random


class Environment:
    def __init__(self, action_size, state_size,
                 action_range, position_range,
                 max_steps=200, threshold=0.5, penalty_term=0, reward_constant=10, verbose=False):
        self._action_size = action_size
        self._state_size = state_size

        self._action_range = action_range
        self._position_range = position_range

        self._max_steps = max_steps
        self._threshold = threshold
        self._penalty_term = penalty_term
        self._reward_constant = reward_constant
        self._verbose = verbose

        """
            TCP server configuration
        """
        self._HOST = '127.0.0.1'
        self._PORT = 30001

        # setup tcp server
        self._server = None
        self._connection = None
        # self._setup_tcp_server()

        # change for every reset
        self._state = None
        self._target = None
        self._current_step = None
        self._prev_error = None

        self.reset()

        self._force = np.zeros((1, self._action_size))

    """
        reset the environment to the initial state
    """

    def reset(self):
        self._state = np.zeros((1, self._state_size))
        self._current_step = 0
        self._prev_error = 0
        self._target = np.array([np.random.uniform(low=self._position_range[0][0], high=self._position_range[0][1]),
                                 np.random.uniform(low=self._position_range[1][0], high=self._position_range[1][1])])

        return self._state, {}

    def step(self, action):
        self._force = [a + b for a, b in zip(self._force, action)]

        self._current_step += 1

        self._state = self._simulate()

        reward = self._calculate_reward()
        done = self._state[4] == 1
        truncated = self._current_step >= self._max_steps

        return self._state, reward, done, truncated

    def _calculate_reward(self):
        error = np.sum(np.sqrt([self._state[2] ** 2, self._state[3] ** 2]))

        reward = None
        if error > self._threshold:
            reward = -self._penalty_term + self._prev_error - error

        if error <= self._threshold:
            reward = self._penalty_term + self._prev_error - error + self._reward_constant

        self._prev_error = error

        return reward

    def _simulate(self):
        return np.array([
            random.uniform(self._position_range[0][0], self._position_range[0][1]),
            random.uniform(self._position_range[1][0], self._position_range[0][1]),
            self._target[0] - random.uniform(self._position_range[0][0], self._position_range[0][1]),
            self._target[1] - random.uniform(self._position_range[0][0], self._position_range[0][1]),
            0
        ])
        pass

        self._send_message()

        return self._receive_message()

    def _send_message(self):
        if self._server is None or self._connection is None:
            raise Exception('No server connected')

        try:
            self._connection.sendall(json.dumps({
                'data': self._force,
            }).encode())

            if self._verbose:
                print(f"Sent, force: {self._force}")
        except Exception as e:
            raise Exception(f"Failed to send message: {e}")

    def _receive_message(self):
        if self._server is None or self._connection is None:
            raise Exception('No server connected')

        data = self._connection.recv(1024)
        if not data:
            raise Exception("Client disconnected.")

        data = json.loads(data.decode())['data']

        self._state = np.array([data[0],
                                data[1],
                                data[0] - self._target[0],
                                data[1] - self._target[1],
                                1 if data[0] == self._target[0] and data[1] == self._target[1] else 0])

        return data

    def _setup_tcp_server(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind((self._HOST, self._PORT))
        self._server.listen(1)
        print(f"Server is listening on {self._HOST}:{self._PORT}")

        self._connection, _ = self._server.accept()
