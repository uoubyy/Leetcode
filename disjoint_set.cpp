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

namespace BFS
{
bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
	vector<vector<int>> neighbours(n);

	for (auto edge : edges)
	{
		neighbours[edge[0]].push_back(edge[1]);
		neighbours[edge[1]].push_back(edge[0]);
	}

	vector<bool> visited(n, false);
	queue<int> nextNodes;
	nextNodes.push(source);

	while (!nextNodes.empty())
	{
		int cur = nextNodes.front();
		nextNodes.pop();

		if (cur == destination)
			return true;

		visited[cur] = true;

		for (auto node : neighbours[cur])
		{
			if (!visited[node])
				nextNodes.push(node);
		}
	}
	return false;
}

vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
	int n = graph.size();
	vector<vector<int>> allPaths;

	queue<vector<int>> curPaths;
	curPaths.push({0});

	vector<bool> visited(n, false);

	while (!curPaths.empty())
	{
		vector<int> cur = curPaths.front();
		curPaths.pop();

		fill(visited.begin(), visited.end(), false);
		for (auto node : cur)
			visited[node] = true;

		for (auto node : graph[cur.back()])
		{
			if (node == n - 1)
			{
				vector<int> path(cur);
				path.push_back(n - 1);
				allPaths.push_back(path);
			}
			else if (!visited[node])
			{
				vector<int> path(cur);
				path.push_back(node);
				curPaths.push(path);
			}
		}
	}

	return allPaths;
}

// Definition for a Node.
class Node {
public:
	int val;
	Node* left;
	Node* right;
	Node* next;

	Node() : val(0), left(NULL), right(NULL), next(NULL) {}

	Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

	Node(int _val, Node* _left, Node* _right, Node* _next)
		: val(_val), left(_left), right(_right), next(_next) {}
};

Node* connect(Node* root) {
	queue<Node*> curLevel;
	queue<Node*> nextLevel;

	curLevel.push(root);

	while (!curLevel.empty())
	{
		Node* cur = curLevel.front();
		curLevel.pop();

		if (cur == nullptr)
			break;
		Node* curRight = curLevel.empty() ? nullptr : curLevel.front();

		cur->right = curRight;

		nextLevel.push(cur->left);
		nextLevel.push(cur->right);

		if (curLevel.empty())
			swap(curLevel, nextLevel);
	}

	return root;
}

struct Point {
	int x;
	int y;
	Point(int _x, int _y) : x(_x), y(_y) {}
};

int shortestPathBinaryMatrix(vector<vector<int>> grid) {
	int n = grid.size();
	vector<vector<bool>> visited(n, vector<bool>(n, false));

	vector<Point> directions{{-1, 0}, {-1, -1}, {0, -1}, {1, -1},
							 {1, 0},  {1, 1},	{0, 1},	 {-1, 1}};

	queue<Point> curLevel;
	queue<Point> nextLevel;

	if (grid[0][0] == 1)
		return -1;
	curLevel.push({0, 0});
	int depth = 1;

	while (!curLevel.empty())
	{
		Point cur = curLevel.front();
		curLevel.pop();
		visited[cur.x][cur.y] = true;

		if (cur.x == n - 1 && cur.y == n - 1)
			return depth;

		for (auto dir : directions)
		{
			int x = cur.x + dir.x;
			int y = cur.y + dir.y;
			if (x < 0 || y < 0 || x >= n || y >= n)
				continue;

			if (!visited[x][y] && grid[x][y] == 0)
			{
				nextLevel.push({x, y});
				visited[x][y] = true;
			}
		}

		if (curLevel.empty())
		{
			swap(curLevel, nextLevel);
			++depth;
		}
	}

	return depth;
}

int orangesRotting(vector<vector<int>> grid) {
	vector<int> direction{-1, 0, 1, 0, -1};

	int cnt = 0;
	int time = 0;
	queue<vector<int>> curLevel;

	int n = grid.size(), m = grid[0].size();
	for (int x = 0; x < n; ++x)
	{
		for (int y = 0; y < m; ++y)
		{
			if (grid[x][y] == 2)
				curLevel.push({x, y});
			else if (grid[x][y] == 1)
				++cnt;
		}
	}

	queue<vector<int>> nextLevel;
	while (!curLevel.empty())
	{
		vector<int> cur = curLevel.front();
		curLevel.pop();
		for (int i = 0; i < 4; ++i)
		{
			int x = cur[0] + direction[i];
			int y = cur[1] + direction[i + 1];
			if (x < 0 || y < 0 || x >= n || y >= m || grid[x][y] != 1)
				continue;

			nextLevel.push({x, y});
			grid[x][y] = 2;
			--cnt;
		}

		if (curLevel.empty() && !nextLevel.empty())
		{
			++time;
			swap(curLevel, nextLevel);
		}
	}

	return cnt == 0 ? time : -1;
}

}  // namespace BFS

namespace MST
{
int FindRoot(vector<int>& roots, int node) {
	int root = roots[node];
	vector<int> path;
	while (root != roots[root])
	{
		path.push_back(root);
		root = roots[root];
	}

	for (auto p : path)
		roots[p] = root;

	roots[node] = root;
	return root;
}

int minCostConnectPoints(vector<vector<int>> points) {
	multimap<int, vector<int>> edges;

	int n = points.size();
	vector<int> roots(n);
	for (int i = 0; i < n; ++i)
		roots[i] = i;

	for (int i = 0; i < n; ++i)
	{
		for (int j = i + 1; j < n; ++j)
			edges.insert(
				pair<int, vector<int>>(abs(points[i][0] - points[j][0]) +
										   abs(points[i][1] - points[j][1]),
									   {i, j}));
	}

	int cost = 0;
	for (auto it = edges.begin(); it != edges.end(); ++it)
	{
		int left = FindRoot(roots, (*it).second[0]);
		int right = FindRoot(roots, (*it).second[1]);

		if (left != right)
		{
			cost += (*it).first;
			roots[(*it).second[1]] = left;
			roots[right] = left;
			--n;
			if (n == 1)
				break;
		}
	}

	return cost;
}

class DSU {
private:
	vector<int> parent, rank;

public:
	DSU(int n) {
		parent.resize(n);
		iota(begin(parent), end(parent), 0);
		rank.resize(n, 1);
	}
	int find_parent(int node) {
		if (node == parent[node])
			return node;
		return parent[node] = find_parent(parent[node]);
	}
	void Union(int u, int v) {
		int U = find_parent(u), V = find_parent(v);
		if (U == V)
			return;
		if (rank[U] < rank[V])
			swap(U, V);
		rank[U] += rank[V];
		parent[V] = U;
	}
};

}  // namespace MST

int main() {
	cout << BFS::shortestPathBinaryMatrix({{0, 0, 0}, {1, 1, 0}, {1, 1, 0}})
		 << endl;

	cout << BFS::orangesRotting({{2, 1, 1}, {1, 1, 1}, {0, 1, 2}}) << endl;

	cout << MST::minCostConnectPoints(
				{{7, 18}, {-15, 19}, {-18, -15}, {-7, 14}, {4, 1}, {-6, 3}})
		 << endl;
	return 0;
}