import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Knowledge Graph
class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, name, description):
        self.nodes[name] = description

    def add_edge(self, node1, node2, description):
        self.edges[(node1, node2)] = description

    def get_node(self, name):
        return self.nodes.get(name, None)

    def get_edge(self, node1, node2):
        return self.edges.get((node1, node2), None)

# Environment
class Environment:
    def __init__(self):
        self.state = "initial"

    def step(self, action):
        if action == "A":
            self.state = "success"
            return 1  # پاداش مثبت
        elif action == "B":
            self.state = "neutral"
            return 0  # بدون پاداش
        else:
            self.state = "fail"
            return -1  # پاداش منفی

# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Decision-Making Module
class DecisionMakingModule:
    def __init__(self, model, optimizer, input_size):
        self.model = model
        self.optimizer = optimizer
        self.input_size = input_size

    def evaluate_options(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(state_tensor)
        return torch.argmax(scores).item()

    def update_model(self, state, action, reward):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.int64)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        # پیش‌بینی
        predicted_scores = self.model(state_tensor)
        predicted_value = predicted_scores.gather(1, action_tensor.unsqueeze(1)).squeeze()

        # محاسبه خطا
        loss = F.mse_loss(predicted_value, reward_tensor)

        # به‌روزرسانی مدل
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Learning Module
class LearningModule:
    def __init__(self, decision_making_module, environment):
        self.decision_making_module = decision_making_module
        self.environment = environment

    def train(self, episodes):
        for episode in range(episodes):
            state = [random.random() for _ in range(self.decision_making_module.input_size)]
            action = self.decision_making_module.evaluate_options(state)
            reward = self.environment.step(["A", "B", "C"][action])
            self.decision_making_module.update_model(state, action, reward)
            print(f"Episode {episode+1}: Action = {action}, Reward = {reward}")

# Main
if __name__ == "__main__":
    # تنظیمات
    input_size = 10  # اندازه ورودی
    output_size = 3  # تعداد اقدامات ممکن (A, B, C)
    episodes = 100  # تعداد اپیزودها

    # گراف دانش
    knowledge_graph = KnowledgeGraph()
    knowledge_graph.add_node("A", "Action A: High Reward")
    knowledge_graph.add_node("B", "Action B: Neutral")
    knowledge_graph.add_node("C", "Action C: Low Reward")

    # محیط
    environment = Environment()

    # شبکه عصبی و ماژول تصمیم‌گیری
    model = NeuralNetwork(input_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    decision_making_module = DecisionMakingModule(model, optimizer, input_size)

    # ماژول یادگیری
    learning_module = LearningModule(decision_making_module, environment)

    # آموزش Agent
    learning_module.train(episodes)

    # آزمایش Agent
    test_state = [random.random() for _ in range(input_size)]
    test_action = decision_making_module.evaluate_options(test_state)
    print(f"Test Action: {test_action}")