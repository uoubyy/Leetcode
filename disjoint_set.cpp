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
class Edge {
public:
	int cost;
	int point1;
	int point2;

	Edge(int _cost, int _p1, int _p2) : cost(_cost), point1(_p1), point2(_p2) {}
};

bool operator<(const Edge& e1, const Edge& e2) {
	return e1.cost > e2.cost;
}

int minCostConnectPoints(vector<vector<int>> points) {
	int n = points.size();
	vector<bool> memo(n, false);
	memo[0] = true;
	vector<int> visited;
	visited.push_back(0);

	vector<priority_queue<Edge>> edges(n);

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if (i != j)
			{
				int cost = abs(points[i][0] - points[j][0]) +
						   abs(points[i][1] - points[j][1]);
				edges[i].push(Edge(cost, i, j));
			}
		}
	}

	int sum = 0;
	while (n > 1)
	{
		int best = INT_MAX;
		int idx = -1;
		for (int i = 0; i < visited.size(); ++i)
		{
			while (!edges[visited[i]].empty())
			{
				Edge edge = edges[visited[i]].top();
				if (memo[edge.point2])
				{ edges[visited[i]].pop(); }
				else
				{
					if (edge.cost < best)
					{
						best = edge.cost;
						idx = edge.point2;
					}
					break;
				}
			}
		}

		sum += best;
		memo[idx] = true;
		visited.push_back(idx);
		--n;
	}

	return sum;
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

namespace SSSP
{
/*
int networkDelayTime(vector<vector<int>> times, int n, int k) {
	vector<int> cost(n + 1, INT_MAX);
	cost[k] = 0;

	int curr = k;
	vector<bool> visited(n + 1, false);
	visited[k] = true;

	vector<vector<int>> edges(n + 1, vector<int>(n + 1, INT_MAX));
	for (auto time : times)
		edges[time[0]][time[1]] = time[2];

	while (true)
	{
		visited[curr] = true;
		for (int i = 1; i <= n; ++i)
		{
			// for all the nodes connected
			// update the shortest path cost
			if (!visited[i] && edges[curr][i] != INT_MAX)
				cost[i] = min(cost[i], cost[curr] + edges[curr][i]);
		}

		// find the next start node
		int best = INT_MAX;

		for (int i = 1; i <= n; ++i)
		{
			if (!visited[i] && cost[i] < best)
			{
				best = cost[i];
				curr = i;
			}
		}

		if (best == INT_MAX)
			break;
	}

	int delay = 0;
	for (int i = 1; i <= n; ++i)
		delay = max(delay, cost[i]);

	return delay == INT_MAX ? -1 : delay;
}
*/
vector<pair<int, int>> adj[101];
void dijkstra(vector<int>& signalReceivedAt, int source, int n) {
	priority_queue<pair<int, int>, vector<pair<int, int>>,
				   greater<pair<int, int>>>
		pq;

	pq.push({0, source});

	// time for starting node is 0
	signalReceivedAt[source] = 0;

	while (!pq.empty())
	{
		int currNodeTime = pq.top().first;
		int currNode = pq.top().second;
		pq.pop();

		if (currNodeTime > signalReceivedAt[currNode])
			continue;

		// broadcast the signal to adjacent nodes
		for (pair<int, int> edge : adj[currNode])
		{
			int time = edge.first;
			int neighborNode = edge.second;

			if (signalReceivedAt[neighborNode] > currNodeTime + time)
			{
				signalReceivedAt[neighborNode] = currNodeTime + time;
				pq.push({signalReceivedAt[neighborNode], neighborNode});
			}
		}
	}
}

int networkDelayTime(vector<vector<int>> times, int n, int k) {
	// build the adjacent list
	for (vector<int> time : times)
	{
		int source = time[0];
		int dest = time[1];
		int travelTime = time[2];

		adj[source].push_back({travelTime, dest});
	}
	vector<int> signalReceivedAt(n + 1, INT_MAX);
	dijkstra(signalReceivedAt, k, n);

	int ans = INT_MAX;
	for (int i = 1; i <= n; ++i)
		ans = max(ans, signalReceivedAt[i]);

	return ans == INT_MAX ? -1 : ans;
}

int findCheapestPrice(
	int n, vector<vector<int>> flights, int src, int dst, int k) {
	k++;
	vector<vector<int>> dp(k + 1, vector<int>(n, INT_MAX));

	for (int i = 0; i <= k; ++i)
		dp[i][src] = 0;

	while (true)
	{
		bool changed = false;
		for (auto flight : flights)
		{
			int c1 = flight[0];
			int c2 = flight[1];
			int price = flight[2];

			for (int i = 1; i <= k; ++i)
			{
				if (dp[i - 1][c1] == INT_MAX)
					continue;

				if (dp[i][c2] > (dp[i - 1][c1] + price))
				{
					changed = true;
					dp[i][c2] = dp[i - 1][c1] + price;
				}
			}
		}
		if (!changed)
			break;
	}

	return dp[k][dst] == INT_MAX ? -1 : dp[k][dst];
}

int findCheapestPriceI(
	int n, vector<vector<int>> flights, int src, int dst, int k) {
	vector<int> curr(n, INT_MAX);
	vector<int> prev(n, INT_MAX);
	prev[src] = 0;
	curr[src] = 0;

	k++;
	while (k--)
	{
		bool changed = false;

		for (auto flight : flights)
		{
			int c1 = flight[0];
			int c2 = flight[1];
			int price = flight[2];

			if (prev[c1] != INT_MAX && prev[c1] + price < curr[c2])
			{
				curr[c2] = prev[c1] + price;
				changed = true;
			}
		}

		swap(prev, curr);

		if (!changed)
			break;
	}

	return prev[dst] == INT_MAX ? -1 : prev[dst];
}

struct Cell {
	bool dirty;
	int effort;

	Cell() : dirty(false), effort(INT_MAX) {}
	Cell(int _x, int _y) : dirty(false), effort(INT_MAX) {}
};

bool operator<(const Cell& left, const Cell& right) {
	return left.effort < right.effort;
}

int minimumEffortPath(vector<vector<int>> heights) {
	int n = heights.size(), m = heights[0].size();

	vector<vector<Cell>> memo(n, vector<Cell>(m));
	memo[0][0].effort = 0;

	queue<vector<int>> cells;
	vector<int> directions{-1, 0, 1, 0, -1};

	cells.push({0, 0});

	while (!cells.empty())
	{
		vector<int> curr = cells.front();
		cells.pop();
		memo[curr[0]][curr[1]].dirty = false;

		for (int i = 0; i < 4; ++i)
		{
			int x = curr[0] + directions[i];
			int y = curr[1] + directions[i + 1];
			if (x < 0 || y < 0 || x >= n || y >= m)
				continue;

			int effort = abs(heights[curr[0]][curr[1]] - heights[x][y]);
			effort = max(memo[curr[0]][curr[1]].effort, effort);

			if (effort < memo[x][y].effort)
			{
				memo[x][y].effort = effort;
				if (!memo[x][y].dirty)
				{
					memo[x][y].dirty = true;
					cells.push({x, y});
				}
			}
		}
	}

	return memo[n - 1][m - 1].effort;
}

struct Course {
	int prevCnt;
	vector<int> children;
};

bool operator<(const Course& left, const Course& right) {
	return left.prevCnt < right.prevCnt;
}

vector<int> findOrder(int numCourses, vector<vector<int>> prerequisites) {
	vector<int> res;

	vector<Course> courses(numCourses);

	for (auto require : prerequisites)
	{
		courses[require[0]].prevCnt++;
		courses[require[1]].children.push_back(require[0]);
	}

	queue<int> currLevel;
	queue<int> nextLevel;
	vector<bool> visited(numCourses, false);

	for (int i = 0; i < numCourses; ++i)
	{
		if (courses[i].prevCnt == 0)
			currLevel.push(i);
	}

	while (!currLevel.empty())
	{
		int id = currLevel.front();
		currLevel.pop();

		Course curr = courses[id];
		visited[id] = true;

		numCourses--;

		res.push_back(id);
		for (auto child : curr.children)
		{
			courses[child].prevCnt--;
			if (courses[child].prevCnt == 0 && !visited[child])
				nextLevel.push(child);
		}

		if (currLevel.empty())

			swap(currLevel, nextLevel);
	}

	if (numCourses != 0)
		return {};

	return res;
}

}  // namespace SSSP

int main() {
	cout << BFS::shortestPathBinaryMatrix({{0, 0, 0}, {1, 1, 0}, {1, 1, 0}})
		 << endl;

	cout << BFS::orangesRotting({{2, 1, 1}, {1, 1, 1}, {0, 1, 2}}) << endl;

	cout << MST::minCostConnectPoints({{0, 0}, {2, 2}, {3, 10}, {5, 2}, {7, 0}})
		 << endl;

	cout << SSSP::networkDelayTime(
				{{2, 4, 10}, {5, 2, 38}, {3, 4, 33}, {4, 2, 76}, {3, 2, 64},
				 {1, 5, 54}, {1, 4, 98}, {2, 3, 61}, {2, 1, 0},	 {3, 5, 77},
				 {5, 1, 34}, {3, 1, 79}, {5, 3, 2},	 {1, 2, 59}, {4, 3, 46},
				 {5, 4, 44}, {2, 5, 89}, {4, 5, 21}, {1, 3, 86}, {4, 1, 95}},
				5, 1)
		 << endl;

	cout << SSSP::findCheapestPriceI(7,
									 {{0, 3, 7},
									  {4, 5, 3},
									  {6, 4, 8},
									  {2, 0, 10},
									  {6, 5, 6},
									  {1, 2, 2},
									  {2, 5, 9},
									  {2, 6, 8},
									  {3, 6, 3},
									  {4, 0, 10},
									  {4, 6, 8},
									  {5, 2, 6},
									  {1, 4, 3},
									  {4, 1, 6},
									  {0, 5, 10},
									  {3, 1, 5},
									  {4, 3, 1},
									  {5, 4, 10},
									  {0, 1, 6}},
									 2, 4, 1)
		 << endl;

	cout << SSSP::minimumEffortPath({{1, 2, 1, 1, 1},
									 {1, 2, 1, 2, 1},
									 {1, 2, 1, 2, 1},
									 {1, 2, 1, 2, 1},
									 {1, 1, 1, 2, 1}})
		 << endl;

	print_vector<int>(SSSP::findOrder(4, {{1, 0}, {2, 0}, {3, 1}, {3, 2}}));
	return 0;
}