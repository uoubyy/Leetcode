#include "env.h"
#include <iostream>

class UnionFind {
public:
	UnionFind(int n) : root(n), rank(n, 1) {
		for (int i = 0; i < n; ++i)
			root[i] = i;
	}

	int FindRoot(int i) {
		vector<int> path;

		int curr = i;
		int parent = root[i];
		while (parent != curr)
		{
			path.push_back(parent);
			curr = parent;
			parent = root[curr];
		}

		for (auto num : path)

			root[num] = curr;

		root[i] = curr;
		return curr;
	}

	void UnionSet(int i, int j) {
		int parent_i = FindRoot(i);
		int parent_j = FindRoot(j);
		if (parent_i != parent_j)
		{
			int rank_i = rank[parent_i];
			int rank_j = rank[parent_j];

			if (rank_i > rank_j)
				root[parent_j] = parent_i;
			else if (rank_i < rank_j)
				root[parent_i] = parent_j;
			else
			{
				root[parent_j] = parent_i;
				rank[parent_i] += 1;
			}
		}
	}

	bool Connected(int i, int j) { return FindRoot(i) == FindRoot(j); }

	int GroupCount() {
		set<int> uniqueRoot;
		int cnt = 0;
		for (int i = 0; i < root.size(); ++i)
		{
			int r = FindRoot(i);
			if (!uniqueRoot.count(r))
			{
				uniqueRoot.insert(r);
				++cnt;
			}
		}

		return cnt;
	}

public:
	vector<int> root;
	vector<int> rank;
};

string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
	int n = s.length();
	UnionFind unionFind(n);
	for (auto info : pairs)
		unionFind.UnionSet(info[0], info[1]);

	map<int, vector<int>> info;
	for (int i = 0; i < n; ++i)
	{
		int root = unionFind.FindRoot(i);
		if (!info.count(root))
			info[root] = vector<int>(26, 0);

		info[root][s[i] - 'a']++;
	}

	for (int i = 0; i < n; ++i)
	{
		int root = unionFind.FindRoot(i);
		for (int j = 0; j < 26; ++j)
		{
			if (info[root][j] > 0)
			{
				s[i] = 'a' + j;
				info[root][j]--;
				break;
			}
		}
	}

	return s;
}

namespace EvaluateDivision
{
unordered_map<string, string> root;
unordered_map<string, double> rates;

string FindRoot(string& str) {
	int n = root.size();
	if (!root.count(str))
	{
		rates[str] = 1.0;
		root[str] = str;
		return str;
	}

	vector<string> path;
	string curr = root[str];
	double multiple = rates[str];

	while (root[curr] != curr)
	{
		path.push_back(curr);
		multiple *= rates[curr];
		curr = root[curr];
	}

	root[str] = curr;

	double prev = rates[str];
	rates[str] = multiple;

	for (int i = 0; i < path.size(); ++i)
	{
		root[path[i]] = curr;
		double tmp = rates[path[i]];
		rates[path[i]] = multiple / prev;
		prev = tmp;
	}

	return curr;
}

vector<double> calcEquation(vector<vector<string>> equations,
							vector<double> values,
							vector<vector<string>> queries) {
	int n = 0;
	for (int i = 0; i < equations.size(); ++i)
	{
		auto equation = equations[i];
		string left = equation[0];
		string right = equation[1];

		string root0 = FindRoot(left);
		string root1 = FindRoot(right);

		if (root0 != root1)
		{
			root[root1] = root0;
			rates[root1] = rates[left] * values[i] / rates[right];
		}
	}

	vector<double> ans;
	for (auto query : queries)
	{
		string left = query[0];
		string right = query[1];
		if (!root.count(left) || !root.count(right))
		{
			ans.push_back(-1.0);
			continue;
		}

		if (FindRoot(left) != FindRoot(right))
			ans.push_back(-1.0);
		else
			ans.push_back(rates[right] / rates[left]);
	}

	return ans;
}
}  // namespace EvaluateDivision

namespace WaterDistribution
{

int FindRoot(vector<int>& costs, vector<int>& roots, int node) {
	int root = roots[node];
	int cost = costs[node];

	vector<int> path;
	path.push_back(node);

	while (roots[root] != root)
	{
		path.push_back(root);
		cost += costs[root];
		root = roots[root];
	}

	int tmp = costs[node];
	for (int i = 0; i < path.size(); ++i)
	{
		tmp = costs[path[i]];
		roots[path[i]] = root;
		costs[path[i]] = cost;
		cost -= tmp;
	}

	return root;
}

int minCostToSupplyWater(int n, vector<int> wells, vector<vector<int>> pipes) {
	vector<int> costs(wells);
	vector<int> roots(n, 0);
	for (int i = 0; i < n; ++i)
		roots[i] = i;

	for (auto pipe : pipes)
	{
		int p1 = pipe[0] - 1;  // start from 0
		int p2 = pipe[1] - 1;

		FindRoot(costs, roots, p1);
		FindRoot(costs, roots, p2);

		if (costs[p1] < pipe[2] && costs[p2] < pipe[2])
		{
			// do nothing
		}
		else if (costs[roots[p2]] > costs[roots[p1]])
		{
			roots[p2] = p1;
			costs[p2] = pipe[2];
		}
		else
		{
			roots[p1] = p2;
			costs[p1] = pipe[2];
		}
	}

	for (int i = 0; i < n; ++i)
		FindRoot(costs, roots, i);

	return accumulate(costs.begin(), costs.end(), 0);
}
}  // namespace WaterDistribution

namespace DFS
{
bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
	stack<int> path;
	vector<bool> visited(n, false);

	unordered_map<int, set<int>> neighbours;
	for (auto edge : edges)
	{
		neighbours[edge[0]].insert(edge[1]);
		neighbours[edge[1]].insert(edge[0]);
	}

	path.push(source);
	while (!path.empty())
	{
		int top = path.top();
		path.pop();
		visited[top] = true;

		if (top == destination)
			return true;

		for (auto neighbor : neighbours[top])
		{
			if (!visited[neighbor])
				path.push(neighbor);
		}
	}

	return false;
}

vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
	int n = graph.size();
	vector<bool> visited(n, false);

	vector<vector<int>> allPath;
	stack<vector<int>> path;
	path.push({0});

	while (!path.empty())
	{
		fill(visited.begin(), visited.end(), false);

		vector<int> top = path.top();
		path.pop();

		int cur = top.back();

		for (auto node : top)
			visited[node] = true;

		bool end = true;
		for (auto node : graph[cur])
		{
			if (!visited[node])
			{
				end = false;
				vector<int> route(top);
				route.push_back(node);

				if (node == n - 1)
					allPath.push_back(route);
				else
					path.push(route);
			}
		}
	}

	return allPath;
}

// Definition for a Node.
class Node {
public:
	int val;
	vector<Node*> neighbors;
	Node() {
		val = 0;
		neighbors = vector<Node*>();
	}
	Node(int _val) {
		val = _val;
		neighbors = vector<Node*>();
	}
	Node(int _val, vector<Node*> _neighbors) {
		val = _val;
		neighbors = _neighbors;
	}
};

Node* cloneGraph(Node* root) {
	if (root == nullptr)
		return nullptr;
	vector<Node*> allNodes(101, nullptr);
	vector<bool> visited(101, false);

	stack<Node*> nodes;
	nodes.push(root);

	while (!nodes.empty())
	{
		Node* cur = nodes.top();
		nodes.pop();
		visited[cur->val] = true;

		if (allNodes[cur->val] == nullptr)
			allNodes[cur->val] = new Node(cur->val);

		for (auto node : cur->neighbors)
		{
			if (!visited[node->val])
			{
				nodes.push(node);
				if (allNodes[node->val] == nullptr)
					allNodes[node->val] = new Node(node->val);
				allNodes[node->val]->neighbors.push_back(allNodes[cur->val]);
				allNodes[cur->val]->neighbors.push_back(allNodes[node->val]);
			}
		}
	}

	return allNodes[root->val];
}

bool dfs(vector<set<int>>& childNodes,
		 vector<bool>& visited,
		 const int& destination,
		 int cur) {
	if (cur == destination)
		return childNodes[cur].empty();
	else if (childNodes[cur].empty())
		return false;

	for (auto node : childNodes[cur])
	{
		if (visited[node])
			return false;

		visited[node] = node != destination;
		if (!dfs(childNodes, visited, destination, node))
			return false;
	}

	visited[cur] = false;
	return true;
}

bool leadsToDestination(int n,
						vector<vector<int>> edges,
						int source,
						int destination) {
	vector<set<int>> childNodes(n);
	vector<bool> visited(n, false);

	for (auto edge : edges)
	{
		// if (edge[0] == edge[1] || edge[0] == destination)
		// 	return false;

		childNodes[edge[0]].insert(edge[1]);
	}

	return dfs(childNodes, visited, destination, source);
}

}  // namespace DFS

int main() {
	return 0;
}