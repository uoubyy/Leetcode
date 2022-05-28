#include "env.h"
// #include "sort.h"
#include "search.h"

#include <nlohmann/json.hpp>

#ifdef __APPLE__
#	include <unistd.h>	 // for linux or Mac
#endif

#define _USE_SMART_PTR
#include "smartptr.h"
#include "weakptr.h"
#include "heap.h"

// #include <glm/glm.hpp>
// #include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtc/type_ptr.hpp>
// #include <glm/gtx/string_cast.hpp>

// #include <Eigen/Dense>

using namespace std;
// using namespace Eigen;

struct Obj {
	char m_char;
	int m_int;
	float m_float;
	short m_short;
};

void print_item(vector<int>& items);

string reverseLeftWords(string s, int n) {
	string head = s.substr(0, n);
	string tail = s.substr(n);
	return tail + head;
}

vector<int> runningSum(vector<int>& nums) {
	vector<int> result;
	int sum = 0;
	for (int i = 0; i < nums.size(); ++i)
	{
		sum += nums[i];
		result.push_back(sum);
	}
	return result;
}

int minPartitions(string n) {
	int max = 0;
	for (int i = 0; i < n.size(); ++i)
		max = (max > (n[i] - '0')) ? max : (n[i] - '0');
	return max;
}

int maxArea(vector<int>& height) {
	int max = 0;
	int n = height.size();
	for (int i = 0, j = n - 1; i < j;)
	{
		int min = height[i] > height[j] ? height[j] : height[i];
		int cur = (j - i) * (min);
		max = max > cur ? max : cur;
		if (height[i] > height[j])
			j--;
		else
			i++;
	}
	return max;
}

int strStr(string haystack, string needle) {
	size_t m = haystack.size();
	size_t n = needle.size();
	if (m < n)
		return -1;
	if (n == 0)
		return 0;

	vector<unsigned int> offset(n, 0);
	unsigned int cur = 0;
	for (size_t i = 0, j = 1; i < n && j < n;)
	{
		if (needle[i] == needle[j])
		{
			i++;
			offset[j] = i;
			j++;
		}
		else
		{
			if (i == 0 || i == offset[i - 1])
			{
				offset[j] = i;
				j++;
			}
			else
			{ i = offset[i - 1]; }
		}
	}
	size_t i = 0, j = 0;
	for (; i < m && j < n;)
	{
		if (haystack[i] == needle[j])
		{
			i++;
			j++;
		}
		else
		{
			if (j == 0)
			{ i++; }
			else
			{ j = offset[j - 1]; }
		}
	}
	// cout << "i " << i << " j " << j << endl;
	if (j == n)	 // find
		return i - n;
	else
		return -1;
}

vector<vector<int>> largeGroupPositions(string s) {
	vector<vector<int>> result;
	char flag = s[0];
	int len = 1;

	size_t i = 1;
	for (; i < s.length(); ++i)
	{
		if (s[i] == flag)
			len++;
		else
		{
			flag = s[i];
			if (len >= 3)
			{
				vector<int> cup{int(i - len), int(i - 1)};
				result.push_back(cup);
				// cout << i - len << "," << i - 1 << endl;
			}
			len = 1;
		}
	}
	if (len >= 3)
	{
		vector<int> cup{int(i - len), int(i - 1)};
		result.push_back(cup);
		print_item(cup);
		// cout << i - len << "," << i - 1 << endl;
	}

	return result;
}

void print_item(vector<int>& items) {
	for (int item : items)
		cout << item << " ";
}

class Node {
public:
	int m_id;
	Node(int id) : m_id(id) { /*cout << "Construct node " << m_id << endl;*/
	}

	bool operator<(const Node& v) const {
		return m_id < v.m_id;
	}
	bool operator==(const Node& v) const {
		return m_id == v.m_id;
	}
	virtual ~Node() {}
};

class Edge {
public:
	int m_from;	 // node Id
	int m_to;
	double m_cost;
	Edge(int from, int to, double cost)
		: m_from(from), m_to(to), m_cost(cost) {}
	void show() const {
		// cout << "From " << m_from << " to " << m_to << " cost " << m_cost
		//     << endl;
	}
	virtual ~Edge() {}
};

class Graph {
public:
	set<Node> m_nodes;
	typedef list<Edge> EdgeList;
	map<int, EdgeList> m_edges;	 // pair id with edge list

	Graph() {}

	void show() {
		for (auto& list : m_edges)
		{
			for (const Edge& edge : list.second)
				edge.show();
		}
	}
	const Node* GetNode(int idx) {
		for (const Node& node : m_nodes)
			if (node.m_id == idx)
				return &node;

		return nullptr;
	}
	void AddNode(int idx) {
		if (GetNode(idx) == nullptr)
		{
			m_nodes.insert(Node(idx));
			EdgeList edges;
			m_edges[idx] = edges;
		}
	}
	void AddEdge(int from, int to, int cost) {
		AddNode(from);
		AddNode(to);
		Edge edge(from, to, cost);
		m_edges[from].push_back(edge);
		// cout<<"Add edge "<<from<<" "<<to<<" "<<cost<<endl;
	}
	int NodesNum() {
		return m_nodes.size();
	}
	virtual ~Graph() { /*cout << "Deconstruct Graph." << endl;*/
	}
};

shared_ptr<Graph> createGraph(vector<vector<int>>& times) {
	shared_ptr<Graph> graph = make_shared<Graph>();

	for (auto info : times)
	{ graph->AddEdge(info[0], info[1], info[2]); }
	return graph;
}

int search(int idx, shared_ptr<Graph> graph, int N) {
	if (N > graph->NodesNum())
		return -1;
	map<int, int> costToNode;
	auto comp = [&costToNode](int a, int b) {
		return costToNode[a] > costToNode[b];
	};

	priority_queue<int, vector<int>, decltype(comp)> newNodes(comp);
	set<int> visited;
	int max = 0;

	newNodes.push(idx);
	costToNode[idx] = 0;

	while (!newNodes.empty())
	{
		// cout << newNodes.size() << endl;
		// for (auto item : costToNode)
		//    cout << item.first << " " << item.second << endl;
		// cout << "-----------------------------" << endl;

		int cur = newNodes.top();
		// cout << "Top item " << cur << " " << costToNode[cur] << endl;
		newNodes.pop();
		visited.insert(cur);
		int curCost = costToNode[cur];

		bool flag = false;
		for (const Edge& edge : graph->m_edges[cur])
		{
			edge.show();
			if (visited.find(edge.m_to) == visited.end())
			{
				auto it = costToNode.find(edge.m_to);
				if (it != costToNode.end())
				{
					it->second = it->second > (curCost + edge.m_cost) ?
									 (curCost + edge.m_cost) :
									 it->second;
					flag = true;
				}
				else
				{
					costToNode[edge.m_to] = curCost + edge.m_cost;
					newNodes.push(edge.m_to);
					// cout << "Find new node " << edge.m_to << endl;
				}
			}
		}

		if (flag)
		{
			priority_queue<int, vector<int>, decltype(comp)> temp(comp);
			while (!newNodes.empty())
			{
				temp.push(newNodes.top());
				newNodes.pop();
			}
			while (!temp.empty())
			{
				newNodes.push(temp.top());
				temp.pop();
			}
		}
	}

	for (auto it = costToNode.begin(); it != costToNode.end(); ++it)
	{ max = max < it->second ? it->second : max; }

	if (visited.size() == graph->NodesNum())
		return max;
	else
		return -1;
}

int networkDelayTime(vector<vector<int>>& times, int N, int K) {
	shared_ptr<Graph> graph = createGraph(times);
	return search(K, graph, N);
}

typedef unsigned char* byte_pointer;

void show_bytes(byte_pointer start, size_t len) {
	size_t i;
	for (i = 0; i < len; ++i)
	{ printf("%.2x ", start[i]); }
	printf("\n");
}

int isLessOrEqual(int x, int y) {
	int tmin = 1 << 31;

	int v1 = !(x ^ tmin);
	int v2 = !!((x - y) & tmin);
	int v3 = !(x - y);
	int v4 = (y ^ tmin);
	printf("%x %x %x %d \n", v1, v2, v3, x - y);

	return (v1 | v2 | v3);
}

int subsetXORSum(vector<int>& nums) {
	int sum = 0;
	return sum;
}

class Container {
	vector<double> lst;

public:
	Container() {}
	Container(initializer_list<double> il) : lst(il) {}
	int size() const {
		return lst.size();
	}
	virtual ~Container() {}
};

class Vector {
private:
	double* elem;
	int sz;

public:
	Vector(int s) : elem(new double[s]), sz(s) {
		for (int i = 0; i < s; ++i)
			elem[i] = 0;
	}
	~Vector() {
		delete[] elem;
	}
	double& operator[](int i) {
		return elem[i];
	}
	int size() const {
		return sz;
	}
};

/*
string findLongestWord(string s, vector<string>& dictionary)
{
	string str;
	if (s.length() == 0)
		return str;
	auto len = s.length();

	for (auto& word : dictionary) {
		auto i = 0;
		auto j = 0;

		while (i < len && j < word.length()) {
			if (s[i] == word[j]) {
				i++;
				j++;
			}
			else {
				while (i < len) {
					i++;
					if (s[i] == word[j]) {
						i++;
						j++;
						break;
					}
				}
			}
		}

		if (j == word.length()) {
			if (str.empty() || str.length() < word.length() ||
				(str.length() == word.length() && (word < str)))
				str = word;
		}
	}
	return str;
}
*/

vector<vector<int>> threeSum(vector<int>& nums) {
	vector<vector<int>> res;

	sort(nums.begin(), nums.end());
	if (nums.empty() || nums[0] > 0)
		return res;

	int n = nums.size();

	for (int first = 0; first < n; ++first)
	{
		if (first > 0 && nums[first] == nums[first - 1])
			continue;

		int third = n - 1;
		int target = -nums[first];
		for (int second = first + 1; second < n; ++second)
		{
			if (second > first + 1 && nums[second] == nums[second - 1])
				continue;

			while (second < third && nums[second] + nums[third] > target)
			{ --third; }

			if (third == second)
				break;

			if (nums[first] + nums[second] + nums[third] == 0)
				res.push_back(
					vector<int>{nums[first], nums[second], nums[third]});
		}
	}
	return res;
}

// 455
int findContentChildren(vector<int>& children, vector<int>& cookies) {
	sort(children.begin(), children.end());
	sort(cookies.begin(), cookies.end());

	int child = 0, cookie = 0;
	while (child < children.size() && cookie < cookies.size())
	{
		if (children[child] <= cookies[cookie])
			child++;
		cookie++;
	}

	return child;
}

// 135
int candy(vector<int>& ratings) {
	vector<int> candies(ratings.size(), 1);
	return 0;
}

// 605
bool canPlaceFlowers(vector<int>& flowerbed, int n) {
	size_t size = flowerbed.size();
	int indicator = 0;
	int max = 0;
	for (size_t i = 0; i < size; ++i)
	{
		if (flowerbed[i] == 1)
		{
			if (indicator != 0)
			{ max--; }
			indicator = 1;
		}
		else
		{
			if (indicator == 0)
			{
				max++;
				indicator = 1;
			}
			else
				indicator = 0;
		}
	}

	return max >= n;
}

// 452
int findMinArrowShots(vector<vector<int>>& points) {
	sort(points.begin(), points.end(),
		 [](vector<int>& a, vector<int>& b) { return a[0] < b[0]; });

	int arrows = 1;
	int indicator = points[0][1];
	int left = points[0][0];
	int right = points[0][1];
	for (auto point : points)
	{
		if (point[0] <= right)
		{
			left = max(left, point[0]);
			right = min(right, point[1]);
		}
		else
		{
			arrows++;
			left = point[0];
			right = point[1];
		}
	}
	return arrows;
}

// 763
vector<int> partitionLabels(string s) {
	map<char, int> char_info;
	for (int i = 0; i < s.size(); ++i)
	{
		if (char_info.find(s[i]) == char_info.end())
			char_info.insert(pair<char, int>(s[i], i));
		else
			char_info[s[i]] = i;
	}

	vector<vector<int>> regions;
	for (int i = 0; i < s.size(); ++i)
	{
		bool inside = false;
		for (vector<int>& reg : regions)
		{
			if (i <= reg[1] && i >= reg[0])
			{
				inside = true;
				reg[0] = min(reg[0], i);
				reg[1] = max(reg[1], char_info[s[i]]);
				break;
			}
		}

		if (inside == false)
			regions.push_back({i, char_info[s[i]]});
	}

	vector<int> len;
	for (auto reg : regions)
	{ len.push_back(reg[1] - reg[0] + 1); }

	return len;
}

// 122
// int maxProfit(vector<int>& prices) {
// 	int profit = 0;
// 	int size = prices.size();

// 	int prev = INT_MAX;
// 	for (int i = 0; i < size; ++i)
// 	{
// 		if (prices[i] > prev)
// 			profit += (prices[i] - prev);
// 		prev = prices[i];
// 	}

// 	return profit;
// }

// 406
vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
	map<int, vector<int>, greater<int>> group;
	for (auto value : people)
	{
		if (group.find(value[0]) == group.end())
			group[value[0]] = {value[1]};
		else
			group[value[0]].push_back(value[1]);
	}

	for (auto it = group.begin(); it != group.end(); ++it)
	{ sort((*it).second.begin(), (*it).second.end()); }

	vector<vector<int>> res;
	for (auto it = group.begin(); it != group.end(); ++it)
	{
		for (auto index : (*it).second)
		{
			vector<int> pair{(*it).first, index};
			res.insert(res.begin() + index, pair);
		}
	}

	return res;
}

// 665
bool checkPossibility(vector<int>& nums) {
	return false;
}

// 167
// vector<int> twoSum(vector<int>& numbers, int target)
// {
// 	vector<int> pair;
// 	size_t size = numbers.size();
// 	for (size_t l = 0, r = size - 1; l < r;) {
// 		if (numbers[l] + numbers[r] == target) {
// 			pair.push_back(l + 1);
// 			pair.push_back(r + 1);
// 			break;
// 		}
// 		else if (numbers[l] + numbers[r] < target) {
// 			l++;
// 		}
// 		else {
// 			r--;
// 		}
// 	}

// 	return pair;
// }

template <typename T>
void print_vector(vector<T> array) {
	cout << "[";
	for (int i = 0; i < array.size(); ++i)
	{
		cout << array[i];
		if (i < array.size() - 1)
			cout << ", ";
	}
	cout << "]" << endl;
}

template <typename T>
void print_matrix(vector<vector<T>> mat) {
	cout << "[";
	for (int i = 0; i < mat.size(); ++i)
	{
		print_vector(mat[i]);
		if (i < mat.size() - 1)
			cout << ", ";
	}
	cout << "]" << endl;
}

// 88
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
	int pos = m-- + n-- - 1;
	while (m >= 0 && n >= 0)
	{ nums1[pos--] = nums1[m] > nums2[n] ? nums1[m--] : nums2[n--]; }
	while (n >= 0)
	{ nums1[pos--] = nums2[n--]; }
}

struct ListNode {
	int val;
	ListNode* next;
	ListNode(int x) : val(x), next(NULL) {}
};

// 142
ListNode* detectCycle(ListNode* head) {
	ListNode *fast = head, *slow = head;
	do
	{
		if (!fast || !fast->next)
			return nullptr;

		fast = fast->next->next;
		slow = slow->next;
	} while (fast != slow);

	fast = head;
	while (fast != slow)
	{
		fast = fast->next;
		slow = slow->next;
	}

	return fast;
}

// 76
string minWindow(string s, string t) {
	vector<int> chars(128, 0);
	vector<bool> flag(128, false);

	for (auto i = 0; i < t.size(); ++i)
	{
		flag[t[i]] = true;
		++chars[t[i]];
	}

	int cnt = 0, l = 0, min_l = 0, min_size = s.size() + 1;
	for (int r = 0; r < s.size(); ++r)
	{
		if (flag[s[r]])
		{
			if (--chars[s[r]] >= 0)
			{ ++cnt; }

			while (cnt == t.size())
			{
				if (r - l + 1 < min_size)
				{
					min_l = l;
					min_size = r - l + 1;
				}
				if (flag[s[l]] && ++chars[s[l]] > 0)
				{ --cnt; }
				++l;
			}
		}
	}

	return min_size > s.size() ? "" : s.substr(min_l, min_size);
}

// 633
bool judgeSquareSum(int c) {
	int min = 0, max = int(sqrt(c));
	for (; min <= max;)
	{
		int res = c - max * max;
		if (res > min * min)
			++min;
		else if (res < min * min)
			--max;
		else
			return true;
	}

	return false;
}

// 680

bool validSubStr(string& s, int l, int r) {
	while (l < r)
	{
		if (s[l] != s[r])
			return false;
		++l;
		--r;
	}
	return true;
}

bool validPalindrome(string s) {
	// vector<char> chars(s.begin(), s.end());
	int size = s.size();
	int l = 0, r = size - 1;

	while (l < r)
	{
		if (s[l] != s[r])
		{ return (validSubStr(s, l + 1, r) || validSubStr(s, l, r - 1)); }

		++l;
		--r;
	}

	return true;
}

// 524
string findLongestWord(string s, vector<string>& dictionary) {
	// sort(dictionary.begin(), dictionary.end(), [](string& a, string b) {
	//     if (a.size() > b.size())
	//         return true;
	//     else if (a.size() == b.size())
	//         return a < b;
	//     else
	//         return false;
	// });
	// print_vector(dictionary);

	int max_len = 0;
	int max_i = -1;
	int index = 0;
	for (auto& word : dictionary)
	{
		// no need to compare
		if (word.size() > s.size() || max_len > word.size())
		{
			++index;
			continue;
		}

		int cnt = 0;
		for (int i = 0, j = 0; i < s.size() && j < word.size();)
		{
			if (s[i] != word[j])
			{ ++i; }
			else
			{
				++i;
				++j;
				cnt++;
			}
		}

		if (cnt == word.size())
		{
			if (cnt > max_len || (cnt == max_len && word < dictionary[max_i]))
			{
				max_len = cnt;
				max_i = index;
			}
		}

		++index;
	}

	return max_i >= 0 ? dictionary[max_i] : "";
}

// 340
int lengthOfLongestSubstringKDistinct(string s, int k) {
	if (k == 0)
		return 0;

	int size = s.size();

	if (k >= size)
		return size;

	map<char, int> c_num;
	int max = 0;
	int l = 0, r = 0;
	for (; l < size && r < size;)
	{
		auto itor = c_num.find(s[r]);
		if (k == 0 && itor == c_num.end())
		{
			max = (r - l) > max ? (r - l) : max;

			for (; l < r; ++l)
			{
				c_num[s[l]]--;
				if (c_num[s[l]] == 0)
				{
					c_num.erase(s[l]);
					l++;
					break;
				}
			}
			k++;
			continue;
		}

		if (itor == c_num.end())
		{
			--k;
			c_num[s[r]] = 1;
		}
		else
		{ c_num[s[r]]++; }

		r++;
	}

	max = (r - l) > max ? (r - l) : max;
	return max;
}

// 3
int lengthOfLongestSubstring(string s) {
	int size = s.size();

	set<char> chars;
	int max = 0;

	int l = 0, r = 0;
	for (; r < size; ++r)
	{
		if (chars.find(s[r]) == chars.end())
		{ chars.insert(s[r]); }
		else
		{
			max = (r - l) > max ? (r - l) : max;
			for (; l < r;)
			{
				if (s[l] == s[r])
				{
					++l;
					break;
				}
				chars.erase(s[l]);
				++l;
			}
		}
	}

	max = (r - l) > max ? (r - l) : max;
	return max;
}

// 159
int lengthOfLongestSubstringTwoDistinct(string s) {
	int n = s.size();
	vector<char> chars(128);
	int k = 2;

	int l = 0, r = 0;
	int max = 0;

	for (; r < n;)
	{
		if (chars[s[r]] == 0)
		{
			if (k == 0)
			{
				max = (r - l > max) ? (r - l) : max;
				while (l < r)
				{
					chars[s[l]]--;
					if (chars[s[l]] == 0)
					{
						++l;
						chars[s[r]] = 1;
						break;
					}
					++l;
				}
			}
			else
			{
				chars[s[r]] = 1;
				--k;
			}
		}
		else
		{ chars[s[r]]++; }
		++r;
	}

	return (max = (r - l > max) ? (r - l) : max);
}

// 239
// need debug
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
	vector<int> max;
	int l = 0, r = 0;

	int max_v = INT_MIN;
	int max_i = 0;

	while (r < nums.size())
	{
		if (r - l < k)
		{
			max_i = nums[r] > max_v ? r : max_i;
			max_v = nums[r] > max_v ? nums[r] : max_v;
		}
		else
		{
			max.push_back(max_v);
			++l;

			if (l < max_i)
			{
				max_i = nums[r] > max_v ? r : max_i;
				max_v = nums[r] > max_v ? nums[r] : max_v;
			}
			else
			{
				max_v = INT_MIN;
				max_i = l;
				for (int i = l; i <= r; ++i)
				{
					max_i = nums[i] > max_v ? i : max_i;
					max_v = nums[i] > max_v ? nums[i] : max_v;
				}
			}
		}

		r++;
	}
	max.push_back(max_v);

	return max;
}

// 69
int mySqrt(int x) {
	if (x == 0)
		return 0;
	int l = 1, r = x;
	int mid, sqrt;

	while (l <= r)
	{
		mid = l + (r - l) / 2;

		sqrt = x / mid;
		if (sqrt == mid)
			return mid;
		else if (sqrt < mid)
			r = mid - 1;
		else
			l = mid + 1;
	}

	return r;
}

// 367
bool isPerfectSquare(int num) {
	if (num == 1)
		return true;

	int l = 1, r = num;

	int mid, sqrt;

	while (l <= r)
	{
		mid = l + (r - l) / 2;
		sqrt = num / mid;

		if (sqrt == mid)
			return sqrt * sqrt == num;
		else if (sqrt < mid)
			r = mid - 1;
		else
			l = mid + 1;
	}

	return r * r == num;
}

// 34
vector<int> searchRange(vector<int>& nums, int target) {
	int n = nums.size();

	if (n == 0 || target < nums[0] || target > nums[n - 1])
	{ return vector<int>(2, -1); }

	vector<int> res;

	int l = 0, r = n - 1;
	int mid;

	while (l <= r)
	{
		mid = l + (r - l) / 2;
		if (nums[mid] == target)
			break;
		else if (nums[mid] < target)
			l = mid + 1;
		else
			r = mid - 1;
	}

	if (nums[mid] != target)
		return vector<int>(2, -1);

	// find first small
	l = 0;
	r = mid;
	int l_mid;
	while (l <= r)
	{
		l_mid = l + (r - l) / 2;

		// always < or ==
		if (nums[l_mid] == target)
			r = l_mid - 1;
		else
			l = l_mid + 1;
	}
	res.push_back(r + 1);

	l = mid;
	r = n - 1;
	int r_mid;
	while (l <= r)
	{
		r_mid = l + (r - l) / 2;

		// always > or ==
		if (nums[r_mid] == target)
			l = r_mid + 1;
		else
			r = r_mid - 1;
	}
	res.push_back(r);
	return res;
}

// 81
bool search(vector<int>& nums, int target) {
	int n = nums.size();

	int l = 0, r = n - 1;

	while (l <= r)
	{
		int mid = l + (r - l) / 2;

		if (nums[mid] == target)
			return true;

		if (nums[l] == nums[mid])
		{ l++; }
		else if (nums[mid] <= nums[r])
		{
			if (target > nums[mid] && target <= nums[r])
				l = mid + 1;
			else
				r = mid - 1;
		}
		else
		{
			if (target >= nums[l] && target < nums[mid])
				r = mid - 1;
			else
				l = mid + 1;
		}
	}

	return false;
}

// 154
int findMin(vector<int>& nums) {
	int n = nums.size();

	int l = 0, r = n - 1;
	int min = nums[0];
	while (l <= r)
	{
		int mid = l + (r - l) / 2;

		if (nums[mid] == nums[l])
		{
			min = std::min(min, nums[l]);
			l++;
		}
		else if (nums[mid] <= nums[r])	// right side is in order and mid is
										// minimum in right side
		{
			min = std::min(min, nums[mid]);
			r = mid - 1;
		}
		else
		{
			min = std::min(min, nums[mid]);
			l = mid + 1;
		}
	}

	return min;
}

// 540
int singleNonDuplicate(vector<int>& nums) {
	int n = nums.size();

	if (n == 1)
		return nums[0];

	int l = 0, r = n - 1;

	while (l <= r)
	{
		if (l == r)
			return nums[l];

		int mid = l + (r - l) / 2;

		if (nums[mid] == nums[mid - 1])
		{
			if (mid % 2 == 0)
				r = mid - 2;
			else
				l = mid + 1;
		}
		else if (nums[mid] == nums[mid + 1])
		{
			if (mid % 2 == 0)
				l = mid + 2;
			else
				r = mid - 1;
		}
		else
			return nums[mid];
	}

	return nums[l];
}

// 4
double findIthofVector(vector<int>& nums, int i) {
	return 0;
}

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
	int m = nums1.size(), n = nums2.size();
	vector<int> index(2);

	index[0] = (m + n - 1) / 2;
	index[1] = (m + n) / 2;

	if (m == 0)
		return (nums2[index[0]] + nums2[index[1]]) / 2.0;
	else if (n == 0)
		return (nums1[index[0]] + nums1[index[1]]) / 2.0;

	int l = 0, r = m;

	while (true)
	{
		index[0] = (l + r) / 2;
		index[1] = (m + n) / 2 - index[0];

		if (nums1[index[0]] <= nums2[index[1] + 1] &&
			nums2[index[1]] <= nums1[index[0] + 1])
		{
			if ((m + n) % 2 == 0)
				return (max(nums1[index[0] - 1], nums2[index[1] - 1]) +
						min(nums1[index[0]], nums2[index[1]])) /
					   2.0;
			else
				return min(nums1[index[0]], nums2[index[1]]);
		}
		else if (nums2[index[1]] > nums1[index[0] + 1])
		{ l = index[0] + 1; }
	}
}

size_t countChar(const string& str, char c) {
	return 0;
}

// 2099
// vector<int> maxSubsequence(vector<int>& nums, int k) {}

// 1825
class MKAverage {
private:
	int m, k;
	vector<int> steam;

public:
	MKAverage(int i_m, int i_k) {
		m = i_m;
		k = i_k;
	}

	void addElement(int num) {
		steam.push_back(num);
	}

	int calculateMKAverage() {
		if (steam.size() < m)
			return -1;

		vector<int> nums(steam.end() - m, steam.end());

		sort(nums.begin(), nums.end());

		int sum = accumulate(nums.begin() + k, nums.end() - k, 0);

		return sum / m - 2 * k;
	}
};

// 347
vector<int> topKFrequent(vector<int>& nums, int k) {
	/*
	if (nums.size() <= k)
		return nums;

	// print_vector(nums);
	unordered_map<int, int> bucket;
	for (auto num : nums) {
		bucket[num]++;
	}

	multimap<int, int, greater<int>> rank;
	for (auto value : bucket) {
		rank.insert(pair<int, int>(value.second, value.first));
	}

	vector<int> res;
	auto it = rank.begin();
	while (k--) {
		res.push_back(it->second);
		it++;
	}

	return res;

	*/
	unordered_map<int, int> counts;
	int max_count = 0;

	for (const int& num : nums)
		max_count = max(max_count, ++counts[num]);

	vector<vector<int>> buckets(max_count + 1);
	for (const auto& p : counts)
		buckets[p.second].push_back(p.first);

	vector<int> ans;
	for (int i = max_count; i >= 0 && ans.size() < k; --i)
	{
		for (const int& num : buckets[i])
		{
			ans.push_back(num);
			if (ans.size() == k)
				break;
		}
	}

	return ans;
}

string frequencySort(string s) {
	unordered_map<char, int> counts;

	for (int i = 0; i < s.size(); ++i)
		counts[s[i]]++;

	priority_queue<pair<int, char>> char_info;

	for (auto value : counts)
	{ char_info.push({value.second, value.first}); }

	string res;
	while (!char_info.empty())
	{
		auto it = char_info.top();
		int cnt = it.first;
		while (cnt--)
		{ res.push_back(it.second); }

		char_info.pop();
	}

	return res;
}

// 75

void quickSort(vector<int>& nums, int left, int right) {
	if (left + 1 >= right)
		return;
	int l = left, r = right - 1;

	int key = nums[left];

	while (l < r)
	{
		while (l < r && nums[r] >= key)
			--r;
		nums[l] = nums[r];

		while (l < r && nums[l] <= key)
			++l;
		nums[r] = nums[l];
	}

	nums[l] = key;
	quickSort(nums, left, l);
	quickSort(nums, l + 1, right);
}

void sortColors(vector<int>& nums) {
	quickSort(nums, 0, nums.size());
}

// deep-first
extern vector<int> direction;  // {-1, 0, 1, 0, -1};

int dfs(vector<vector<int>>& grid, int r, int c);

int maxAreaOfIsland_1(vector<vector<int>>& grid) {
	int m = grid.size(), n = m ? grid[0].size() : 0;
	//, local_area, area = 0, x, y;
	/*
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (grid[i][j]) {
					local_area = 1;
					grid[i][j] = 0;

					stack<pair<int, int>> island;
					island.push({i, j});

					while (!island.empty()) {
						auto [r, c] = island.top();
						island.pop();

						for (int k = 0; k < 4; ++k) {
							x = r + direction[k];
							y = c + direction[k + 1];

							if (x >= 0 && x < m && y >= 0 && y < n &&
								grid[x][y] == 1) {
								grid[x][y] = 0;
								++local_area;
								island.push({x, y});
							}
						}
					}

					area = max(area, local_area);
				}
			}
		}
	*/

	int max_area = 0;
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if (grid[i][j])
				max_area = max(max_area, dfs(grid, i, j));
		}
	}
	return max_area;
}

int dfs(vector<vector<int>>& grid, int r, int c) {
	if (grid[r][c] == 0)
		return 0;

	grid[r][c] = 0;

	int x, y, area = 1;
	for (int i = 0; i < 4; ++i)
	{
		x = r + direction[i], y = c + direction[i + 1];
		if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size())
			area += dfs(grid, x, y);
	}

	return area;
}

// 547
int findCircleNum_1(vector<vector<int>>& isConnected) {
	int n = isConnected.size();

	int stateNum = 0;
	for (int i = 0; i < n; ++i)
	{
		for (int j = i; j < n; ++j)
		{
			if (isConnected[i][j])
			{
				isConnected[i][j] = 0;

				stateNum++;

				stack<int> cities;
				cities.push(j);

				while (!cities.empty())
				{
					int c = cities.top();
					cities.pop();

					for (int k = 0; k < n; ++k)
					{
						if (isConnected[c][k])
						{
							isConnected[c][k] = 0;
							cities.push(k);
						}
					}
				}
			}
		}
	}

	return stateNum;
}

// 547 better
void dfs(vector<vector<int>>& isConnected, vector<bool>& visited, int i) {
	visited[i] = true;

	for (int k = 0; k < isConnected.size(); ++k)
	{
		if (isConnected[i][k] && !visited[k])
			dfs(isConnected, visited, k);
	}
}

int findCircleNum(vector<vector<int>>& isConnected) {
	int n = isConnected.size(), count = 0;

	vector<bool> visited(n, false);

	for (int i = 0; i < n; ++i)
	{
		if (!visited[i])
		{
			dfs(isConnected, visited, i);
			++count;
		}
	}

	return count;
}

// 417
void dfs(const vector<vector<int>>& matrix,
		 vector<vector<bool>>& can_reach,
		 int r,
		 int c) {
	if (can_reach[r][c])
		return;

	can_reach[r][c] = true;

	int x, y;
	for (int i = 0; i < 4; ++i)
	{
		x = r + direction[i], y = c + direction[i + 1];

		if (x >= 0 && x < matrix.size() && y >= 0 && y < matrix[0].size() &&
			matrix[r][c] <= matrix[x][y])
			dfs(matrix, can_reach, x, y);
	}
}

vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
	if (matrix.empty() || matrix[0].empty())
		return {};

	vector<vector<int>> res;

	int m = matrix.size(), n = matrix[0].size();

	vector<vector<bool>> can_reach_p(m, vector<bool>(n, false));
	vector<vector<bool>> can_reach_a(m, vector<bool>(n, false));

	for (int i = 0; i < m; ++i)
	{
		dfs(matrix, can_reach_p, i, 0);
		dfs(matrix, can_reach_a, i, n - 1);
	}

	for (int i = 0; i < n; ++i)
	{
		dfs(matrix, can_reach_p, 0, i);
		dfs(matrix, can_reach_a, m - 1, i);
	}

	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if (can_reach_a[i][j] && can_reach_p[i][j])
				res.push_back(vector<int>{i, j});
		}
	}

	return res;
}

// 47
void backtracking(vector<int>& nums, int level, vector<vector<int>>& ans) {
	if (level == nums.size() - 1)
	{
		ans.push_back(nums);
		return;
	}

	for (int i = level; i < nums.size();)
	{
		if (i != level && nums[i] == nums[i - 1] && nums[i] == nums[level])
			continue;

		swap(nums[i], nums[level]);
		backtracking(nums, level + 1, ans);
		swap(nums[i], nums[level]);

		do
		{ i++; } while (i < nums.size() && nums[i] != nums[i - 1]);
	}
}
vector<vector<int>> permute(vector<int>& nums) {
	sort(nums.begin(), nums.end());

	vector<vector<int>> res;
	backtracking(nums, 0, res);

	return res;
}

// 77
void backtracking(vector<vector<int>>& res,
				  vector<int>& tmp,
				  int n,
				  int k,
				  int& level,
				  int pos) {
	if (level == k)
	{
		res.push_back(tmp);
		return;
	}

	for (int i = pos; i < n; ++i)
	{
		tmp[level++] = i + 1;
		backtracking(res, tmp, n, k, level, i + 1);
		level--;
	}
}

vector<vector<int>> combine(int n, int k) {
	vector<vector<int>> res;
	vector<int> tmp(k);
	int level = 0;
	backtracking(res, tmp, n, k, level, 0);
	return res;
}

// 79
void backtracking(vector<vector<char>>& board,
				  const string& word,
				  int i,
				  vector<vector<bool>>& visited,
				  int r,
				  int c,
				  bool& find) {
	if (visited[r][c] || board[r][c] != word[i])
		return;

	visited[r][c] = true;

	if (i + 1 == word.size())
	{
		find = true;
		return;
	}

	for (int k = 0; k < 4; ++k)
	{
		int x = r + direction[k], y = c + direction[k + 1];

		if (x < 0 || x >= board.size() || y < 0 || y >= board[0].size())
			continue;

		i++;

		backtracking(board, word, i, visited, x, y, find);
		if (find)
			return;
		i--;
	}

	visited[r][c] = false;
}

bool exist(vector<vector<char>>& board, string word) {
	if (board.empty() || board[0].empty())
		return false;
	vector<vector<bool>> visited(board.size(),
								 vector<bool>(board[0].size(), false));
	bool find = false;

	for (int x = 0; x < board.size(); ++x)
	{
		for (int y = 0; y < board[0].size(); ++y)
		{
			backtracking(board, word, 0, visited, x, y, find);
			if (find)
				return true;
		}
	}

	return find;
}

// 51

void backtracking(vector<vector<string>>& res,
				  vector<string>& board,
				  vector<bool>& column,
				  vector<bool>& ldiag,
				  vector<bool>& rdiag,
				  int row,
				  int n) {
	if (row == n)
	{
		res.push_back(board);
		return;
	}

	for (int i = 0; i < n; ++i)
	{
		if (column[i] || ldiag[n - row + i - 1] || rdiag[row + i])
			continue;

		board[row][i] = 'Q';
		column[i] = ldiag[n - row + i - 1] = rdiag[row + i] = true;

		backtracking(res, board, column, ldiag, rdiag, row + 1, n);

		board[row][i] = '.';
		column[i] = ldiag[n - row + i - 1] = rdiag[row + 1] = false;
	}
}

vector<vector<string>> solveNQueens(int n) {
	vector<vector<string>> res;
	if (n == 0)
		return res;
	vector<string> board(n, string(n, '.'));
	vector<bool> column(n, false), ldiag(2 * n - 1, false),
		rdiag(2 * n - 1, false);

	backtracking(res, board, column, ldiag, rdiag, 0, n);
	return res;
}

// 934
void dfs(queue<pair<int, int>>& points,
		 vector<vector<int>>& grid,
		 int m,
		 int n,
		 int i,
		 int j) {
	if (i < 0 || j < 0 || i == m || j == n || grid[i][j] == 2)
		return;

	if (grid[i][j] == 0)
	{
		points.push({i, j});
		return;
	}

	grid[i][j] = 2;
	dfs(points, grid, m, n, i - 1, j);
	dfs(points, grid, m, n, i, j - 1);
	dfs(points, grid, m, n, i + 1, j);
	dfs(points, grid, m, n, i, j + 1);
}

int shortestBridge(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size();

	queue<pair<int, int>> points;

	bool flipped = false;
	for (int i = 0; i < m; ++i)
	{
		if (flipped)
			break;

		for (int j = 0; j < n; ++j)
		{
			if (grid[i][j] == 1)
			{
				dfs(points, grid, m, n, i, j);	// find all points of island 1
				flipped = true;
				break;
			}
		}
	}

	int x, y;
	int level = 0;

	while (!points.empty())
	{
		++level;

		int n_points = points.size();
		while (n_points--)
		{
			auto [r, c] = points.front();
			points.pop();

			for (int k = 0; k < 4; ++k)
			{
				x = r + direction[k];
				y = c + direction[k + 1];

				if (x >= 0 && y >= 0 && x < m && y < n)
				{
					if (grid[x][y] == 2)  // visited
						continue;

					else if (grid[x][y] == 1)
						return level;

					points.push({x, y});
					grid[x][y] = 2;
				}
			}
		}
	}

	return 0;
}

// 475
// int findRadius(vector<int>& houses, vector<int>& heaters) {

// }

// 126
bool isWorldAdjacent(const string& w1, const string& w2) {
	int m = w1.size(), n = w2.size();
	if (m != n)
		return false;

	int cnt = 0;
	for (int i = 0; i < m; ++i)
	{
		if (w1[i] != w2[i])
			cnt++;

		if (cnt >= 2)
			return false;
	}

	return cnt == 1;
}

void backtracking(const string& src,
				  const string& dst,
				  unordered_map<string, vector<string>>& next,
				  vector<string>& path,
				  vector<vector<string>>& res) {
	if (src == dst)
	{
		res.push_back(path);
		return;
	}

	for (const auto& s : next[src])
	{
		path.push_back(s);
		backtracking(s, dst, next, path, res);
		path.pop_back();
	}
}

vector<vector<string>>
findLadders(string beginWord, string endWord, vector<string>& wordList) {
	vector<vector<string>> res;

	unordered_set<string> dict;

	for (const auto& w : wordList)
		dict.insert(w);

	if (!dict.count(endWord))
		return res;

	dict.erase(beginWord);
	dict.erase(endWord);

	unordered_set<string> q1{beginWord}, q2{endWord};
	unordered_map<string, vector<string>> next;

	bool reversed = false, found = false;

	while (!q1.empty())
	{
		unordered_set<string> q;

		for (const auto& w : q1)
		{
			string s = w;
			for (size_t i = 0; i < s.size(); i++)
			{
				char ch = s[i];
				for (int j = 0; j < 26; j++)
				{
					s[i] = j + 'a';
					if (q2.count(s))  // adjacent word
					{
						reversed ? next[s].push_back(w) : next[w].push_back(s);
						found = true;
					}
					if (dict.count(s))
					{
						reversed ? next[s].push_back(w) : next[w].push_back(s);
						q.insert(s);
					}
				}
				s[i] = ch;	// recover word
			}
		}

		if (found)
		{ break; }

		for (const auto& w : q)
		{ dict.erase(w); }
		if (q.size() <= q2.size())
			q1 = q;
		else
		{
			reversed = !reversed;
			q1 = q2;
			q2 = q;
		}
	}

	if (found)
	{
		vector<string> path = {beginWord};
		backtracking(beginWord, endWord, next, path, res);
	}
	return res;
}

// 130

void backtracking(vector<vector<char>>& board,
				  vector<vector<bool>>& visited,
				  vector<pair<int, int>>& points,
				  char target,
				  int r,
				  int c) {
	if (visited[r][c] || board[r][c] != target)
		return;

	visited[r][c] = true;
	points.push_back(pair<int, int>{r, c});

	for (int i = 0; i < 4; i++)
	{
		int x = r + direction[i], y = c + direction[i + 1];

		if (x >= 0 && y >= 0 && x < board.size() && y < board[0].size())
		{ backtracking(board, visited, points, target, x, y); }
	}
}

void solve(vector<vector<char>>& board) {
	int m = board.size(), n = board[0].size();
	vector<vector<bool>> visited(m, vector<bool>(n, false));

	vector<pair<int, int>> points;

	for (int i = 0; i < m; ++i)
	{
		backtracking(board, visited, points, 'O', i, 0);
		backtracking(board, visited, points, 'O', i, n - 1);
	}

	for (int i = 0; i < n; ++i)
	{
		backtracking(board, visited, points, 'O', 0, i);
		backtracking(board, visited, points, 'O', m - 1, i);
	}

	points.clear();

	for (int i = 1; i < m - 1; i++)
	{
		for (int j = 1; j < n - 1; j++)
		{ backtracking(board, visited, points, 'O', i, j); }
	}

	for (const auto& point : points)
	{ board[point.first][point.second] = 'X'; }
}

void backtracking(TreeNode* node, vector<int>& route, vector<string>& res) {
	route.push_back(node->val);
	if (node->left == nullptr && node->right == nullptr)
	{
		string str;
		for (int i = 0; i < route.size(); i++)
		{
			str += to_string(route[i]);
			if (i < route.size() - 1)
			{ str += "->"; }
		}
		res.push_back(str);
	}

	if (node->left)
		backtracking(node->left, route, res);
	if (node->right)
		backtracking(node->right, route, res);
	route.pop_back();
}

vector<string> binaryTreePaths(TreeNode* root) {
	vector<int> route;
	vector<string> res;
	backtracking(root, route, res);
	return res;
}

void backtracking(map<int, int>& numCount,
				  map<int, int>& numUsed,
				  vector<vector<int>>& res,
				  map<int, int>::iterator itor,
				  int target,
				  int level) {
	if (itor->first > target)
		return;

	if (itor->first == target)
	{
		vector<int> combine;
		for (auto it : numUsed)
		{
			if (it.second > 0)
			{ combine.insert(combine.end(), it.second, it.first); }
		}
		combine.push_back(itor->first);
		res.push_back(combine);

		return;
	}

	int i = level == 0 ? 1 : 0;
	for (; i <= itor->second; ++i)
	{
		if (itor->first * i > target)
			break;

		numUsed[itor->first] = i;
		int tag = target - i * itor->first;

		if (tag == 0)
		{
			vector<int> combine;
			for (auto it : numUsed)
			{
				if (it.second > 0)
				{ combine.insert(combine.end(), it.second, it.first); }
			}

			res.push_back(combine);
			break;
		}

		itor++;
		if (itor == numCount.end())
		{ continue; }
		backtracking(numCount, numUsed, res, itor, tag, ++level);

		itor--;
	}

	numUsed[itor->first] = 0;
}

vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
	vector<vector<int>> res;

	map<int, int> numCount;
	map<int, int> numUsed;
	for (auto val : candidates)
	{
		if (numCount.find(val) != numCount.end())
		{ numCount[val]++; }
		else
		{
			numCount[val] = 1;
			numUsed[val] = 0;
		}
	}

	for (auto itor = numCount.begin(); itor != numCount.end(); itor++)
	{ backtracking(numCount, numUsed, res, itor, target, 0); }

	return res;
}

// 1
vector<int> twoSum(vector<int>& nums, int target) {
	map<int, int> existNum;	 // val, index
	vector<int> res;

	for (int i = 0; i < nums.size(); ++i)
	{
		if (existNum.count(target - nums[i]))
		{
			res.push_back(existNum[target - nums[i]]);
			res.push_back(i);
			break;
		}
		else
		{ existNum[nums[i]] = i; }
	}

	return res;
}

// 310
int backtracking(map<int, vector<int>>& graph, int node, int parent) {
	if (graph.count(node) == 0)
	{ return 1; }

	int height = 0;
	for (auto next : graph[node])
	{
		if (next == parent)
			continue;

		height = max(height, backtracking(graph, next, node));
	}

	return height + 1;
}
vector<int> findMinHeightTreesBFS(int n, vector<vector<int>>& edges) {
	map<int, vector<int>> graph;

	for (auto edge : edges)
	{
		if (graph.count(edge[0]))
			graph[edge[0]].push_back(edge[1]);
		else
			graph[edge[0]] = vector<int>{edge[1]};

		if (graph.count(edge[1]))
			graph[edge[1]].push_back(edge[0]);
		else
			graph[edge[1]] = vector<int>{edge[0]};
	}

	vector<int> res;

	int min_level = INT_MAX;
	for (int i = 0; i < n; i++)
	{
		int level = 1;
		queue<int> curNodes;
		queue<int> nextNodes;
		vector<bool> visited(n, false);

		curNodes.push(i);

		while (!curNodes.empty())
		{
			int node = curNodes.front();
			curNodes.pop();
			visited[node] = true;

			for (auto val : graph[node])
			{
				if (visited[val] == false)
					nextNodes.push(val);
			}

			if (curNodes.empty())
			{
				level++;
				swap(curNodes, nextNodes);
			}
		}

		if (level < min_level)
		{
			res.clear();
			min_level = level;
			res.push_back(i);
		}
		else if (level == min_level)
		{ res.push_back(i); }
	}

	return res;
}

vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
	map<int, vector<int>> graph;

	for (auto edge : edges)
	{
		if (graph.count(edge[0]))
			graph[edge[0]].push_back(edge[1]);
		else
			graph[edge[0]] = vector<int>{edge[1]};

		if (graph.count(edge[1]))
			graph[edge[1]].push_back(edge[0]);
		else
			graph[edge[1]] = vector<int>{edge[0]};
	}

	vector<int> res;

	int min = INT_MAX;
	for (int i = 0; i < n; ++i)
	{
		int h = backtracking(graph, i, i);
		if (h < min)
		{
			min = h;
			res.clear();
			res.push_back(i);
		}
		else if (h == min)
		{ res.push_back(i); }
	}
	return res;
}

bool isPalindrome(int x) {
	if (x < 0)
		return false;

	queue<int> nums;
	int num = x;
	do
	{
		nums.push(num % 10);
		num /= 10;
	} while (num != 0);

	while (!nums.empty())
	{
		num = num * 10 + nums.front();
		nums.pop();
	}

	return num == x;

	// offical solution
	if (x < 0 || (x % 10 == 0 && x != 0))
		return false;

	int revertedNumber = 0;
	while (x > revertedNumber)
	{
		revertedNumber = revertedNumber * 10 + x % 10;
		x /= 10;
	}

	return x == revertedNumber || x == revertedNumber / 10;

	// change into string
	string originNum = to_string(x);
	string reversedNum;
	for (int i = originNum.length() - 1; i >= 0; --i)
		reversedNum.push_back(originNum[i]);

	return originNum == reversedNum;
}

int romanToInt(string s) {
	unordered_map<string, int> str2num;
	str2num["I"] = 1;
	str2num["V"] = 5;
	str2num["X"] = 10;
	str2num["L"] = 50;
	str2num["C"] = 100;
	str2num["D"] = 500;
	str2num["M"] = 1000;
	str2num["IV"] = 4;
	str2num["IX"] = 9;
	str2num["XL"] = 40;
	str2num["XC"] = 90;
	str2num["CD"] = 400;
	str2num["CM"] = 900;

	int sum = 0;
	for (int i = 0; i < s.size();)
	{
		if (i + 1 < s.size() && str2num.count(s.substr(i, 2)))
		{
			sum += str2num[s.substr(i, 2)];
			i += 2;
		}
		else
		{
			sum += str2num[s.substr(i, 1)];
			i += 1;
		}
	}

	return sum;
}

string longestCommonPrefix(vector<string>& strs) {
	int i = 0;
	while (true)
	{
		if (i >= strs[0].size())
			break;
		char tag = strs[0][i];
		for (auto& str : strs)
		{
			if (i >= str.size())
				break;

			if (str[i] != tag)
				return strs[0].substr(0, i);
		}
		i++;
	}

	return strs[0].substr(0, i);
}

bool isValid(string s) {
	stack<char> bucket;
	map<char, char> couple;
	couple[')'] = '(';
	couple['}'] = '{';
	couple[']'] = '[';

	for (int i = 0; i < s.size(); ++i)
	{
		if (!bucket.empty() && (s[i] == ')' || s[i] == '}' || s[i] == ']') &&
			bucket.top() == couple[s[i]])
			bucket.pop();
		else
			bucket.push(s[i]);
	}

	return bucket.empty();
}

// struct ListNode
// {
// 	int val;
// 	ListNode* next;
// 	ListNode() : val(0), next(nullptr) {}
// 	ListNode(int x) : val(x), next(nullptr) {}
// 	ListNode(int x, ListNode* next) : val(x), next(next) {}
// };

ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
	if (list1 == nullptr || list2 == nullptr)
		return list1 == nullptr ? list2 : list1;

	ListNode* head = list1->val < list2->val ? list1 : list2;

	ListNode* l1 = list1->val < list2->val ? list1->next : list1;
	ListNode* l2 = list1->val < list2->val ? list2 : list2->next;
	ListNode* cur = head;

	while (l1 != nullptr && l2 != nullptr)
	{
		if (l1->val < l2->val)
		{
			cur->next = l1;
			cur = cur->next;
			l1 = l1->next;
		}
		else
		{
			cur->next = l2;
			cur = cur->next;
			l2 = l2->next;
		}
	}

	if (l1)
		cur->next = l1;
	if (l2)
		cur->next = l2;

	return head;
}

bool carPooling(vector<vector<int>>& trips, int capacity) {
	// sort(trips.begin(), trips.end(),
	// 	 [](vector<int>& a, vector<int>& b) { return a[1] < b[1]; });

	// int n = trips.size();
	// vector<int> sum(trips[n - 1][2] + 1, 0);
	// for (auto trip : trips) {
	// 	for (int i = trip[1]; i < trip[2]; ++i) {
	// 		sum[i] += trip[0];

	// 		if (sum[i] > capacity) return false;
	// 	}
	// }

	map<int, int> timestamp;

	for (auto trip : trips)
	{
		timestamp[trip[1]] += trip[0];
		timestamp[trip[2]] -= trip[0];
	}

	for (auto itor = timestamp.begin(); itor != timestamp.end(); itor++)
	{
		capacity -= itor->second;
		if (capacity < 0)
			return false;
	}

	return true;
}

void backtracking(string& str, string& max, int l, int r) {
	while (l < r && str[l] == str[r])
	{
		max.push_back(str[l]);
		l++;
		r--;
	}

	if (l > r)
	{
		string tmp = max;
		reverse(max.begin(), max.end());
		tmp += max;
		max = tmp;
		return;
	}
	else if (l == r)
	{
		max.push_back(str[l]);
		string tmp = max;
		max = max.substr(0, max.size() - 1);
		reverse(max.begin(), max.end());
		tmp += max;
		max = tmp;
		return;
	}

	max.clear();

	string m1, m2;
	backtracking(str, m1, l, --r);
	backtracking(str, m2, ++l, r);

	max = m1.size() > m2.size() ? m1 : m2;
}

// string longestPalindrome(string s)
// {
// 	string max;

// 	backtracking(s, max, 0, s.size() - 1);

// 	return max;
// }

constexpr unsigned short middle(int val) {
	static_assert(sizeof(int) == 4, "Unexpected int size.");
	static_assert(sizeof(short) == 2, "Unexpected short size.");

	return (val >> 8) & 0xFFFF;
}

// 70
int getClimbStep(int n, map<int, int>& steps) {
	if (steps.count(n) == 0)
		steps[n] = getClimbStep(n - 1, steps) + getClimbStep(n - 2, steps);
	return steps[n];
}

int climbStairs(int n) {
	map<int, int> steps;
	steps[0] = 0;
	steps[1] = 1;
	steps[2] = 2;

	if (n <= 2)
		return steps[n];

	return getClimbStep(n - 1, steps) + getClimbStep(n - 2, steps);
}

// 198
int backtracking(vector<int>& nums, vector<int>& curMax, int l, int r) {
	if (l >= r)
		return 0;

	if (curMax[l] < 0)
		curMax[l] = max(backtracking(nums, curMax, l + 1, r),
						nums[l] + backtracking(nums, curMax, l + 2, r));

	return curMax[l];
}

int rob(vector<int>& nums) {
	// vector<int> curMax(nums.size(), -1);
	// return backtracking(nums, curMax, 0, nums.size());

	int n = nums.size();
	if (n <= 0)
		return 0;

	vector<int> dp(n + 1, 0);
	dp[1] = nums[0];
	for (int i = 2; i <= n; ++i)
	{ dp[i] = max(dp[i - 1], nums[i - 1] + dp[i - 2]); }

	return dp[n];
}
vector<vector<int>> subarray;
int numberOfArithmeticSlices(vector<int>& nums) {
	int n = nums.size();
	if (n < 3)
		return 0;
	int subs = 0;

	int l = n - 3, r = n - 1;
	int diff = 0;
	while (true)
	{
		diff = nums[r] - nums[r - 1];
		if (diff == nums[l + 1] - nums[l])
		{
			if (l == 0)
			{
				int len = r - l + 1;
				if (len >= 3)
				{
					subs += (len - 1) * (len - 2) / 2;
					vector<int> tmp(nums.begin() + l, nums.begin() + r + 1);
					subarray.push_back(tmp);
				}
				break;
			}
			else
			{ --l; }
		}
		else
		{
			int len = r - l;
			if (len >= 3)
			{
				subs += (len - 1) * (len - 2) / 2;
				vector<int> tmp(nums.begin() + l + 1, nums.begin() + r + 1);
				subarray.push_back(tmp);
			}

			r = l;
			if (len <= 3)
				--l;
		}
	}

	return subs;
}

int minJumps(vector<int>& arr) {
	int n = arr.size();
	vector<bool> visited(n, false);
	map<int, int> idx;

	for (int i = 0; i < n; ++i)
	{ idx[arr[i]] = i; }

	queue<int> curLevel;
	queue<int> nextLevel;
	curLevel.push(0);
	visited[0] = true;

	int level = 0;

	while (!curLevel.empty())
	{
		int index = curLevel.front();
		curLevel.pop();

		if (index == n - 1)
			break;

		if (index + 1 < n && !visited[index + 1])
		{
			nextLevel.push(index + 1);
			visited[index + 1] = true;
		}

		if (index - 1 >= 0 && !visited[index - 1])
		{
			nextLevel.push(index - 1);
			visited[index - 1] = true;
		}

		int val = idx[arr[index]];
		if (!visited[val])
		{
			nextLevel.push(val);
			visited[val] = true;
		}

		if (curLevel.empty())
		{
			++level;
			swap(curLevel, nextLevel);
		}
	}

	return level;
}

class Object {
public:
	int value;

	virtual string getClassName() const {
#ifdef _MSC_VER
		return typeid(*this).name + 6;
#else
		const char* name = typeid(*this).name();

		while (*name >= '0' && *name <= '9')
			name++;

		return name;
#endif
	}

	virtual const type_info& GetTypeInfo() const {
		return typeid(this);
	}
};

class Pawn : public Object {
public:
	virtual string getClassName() const {
		return "Pawn is Child Class.";
	}
};

class Foo {
public:
	void first() {
		printf("AAA\n");

		cout << "first thread id " << this_thread::get_id() << endl;
		// lock2.unlock();
	}

	void second() {
		// lock2.lock();
		printf("BBB\n");

		cout << "second thread id " << this_thread::get_id() << endl;
		// lock3.unlock();
	}

	void third() {
		// lock3.lock();
		printf("CCC\n");

		cout << "third thread id " << this_thread::get_id() << endl;
		// lock3.unlock();
	}

	Foo() {
		// lock2.lock();
		// lock3.lock();
	}

private:
	mutex lock2, lock3;
};

// int subarrayBitwiseORs(vector<int>& arr)
// {
// 	int n = arr.size();
// 	vector<vector<int>> dp(n, vector<int>(n, 1));
// 	unordered_set ans(arr.begin(), arr.end());

// 	for (int i = 0; i < n; ++i) {
// 		for (int j = i; j < n; ++j) {
// 			if (i == j)
// 				dp[i][j] = arr[i];
// 			else
// 				dp[i][j] = dp[i][j - 1] | arr[j];
// 			ans.insert(dp[i][j]);
// 		}
// 	}

// 	return ans.size();
// }

// 898
int subarrayBitwiseORs(vector<int>& arr) {
	unordered_set<int> ans;
	unordered_set<int> cur;
	unordered_set<int> nxt;

	for (auto a : arr)
	{
		nxt.clear();
		nxt.insert(a);

		for (auto b : cur)
			nxt.insert(a | b);

		cur.swap(nxt);
		ans.insert(cur.begin(), cur.end());
	}

	return ans.size();
}

string longestPalindrome(string s) {
	int n = s.size();

	vector<vector<int>> dp(n, vector<int>(n, 0));

	for (int i = 0; i < n; ++i)
		dp[i][i] = 1;

	int len = 1;
	int left = 0;

	for (int i = n - 2; i >= 0; --i)
	{
		for (int j = i + 1; j < n; ++j)
		{
			if (s[i] == s[j])
			{
				if (j == i + 1)
					dp[i][j] = 2;
				else
					dp[i][j] = dp[i + 1][j - 1] > 0 ? dp[i + 1][j - 1] + 2 : 0;

				if (dp[i][j] >= len)
				{
					len = j - i + 1;
					left = i;
				}
			}
			else
			{ dp[i][j] = 0; }
		}
	}

	return s.substr(left, len);
}

bool backtracking(int pos, vector<int>& nums) {
	int n = nums.size() - 1;
	if (pos == n)
		return true;

	int far = min(pos + nums[pos], n);
	for (int i = pos + 1; i <= far; ++i)
	{
		if (backtracking(i, nums))
			return true;
	}

	return false;
}

enum Index { GOOD, BAD, UNKNOW };

bool canJump(vector<int>& nums) {
	int n = nums.size();
	// int far = nums[0];

	// for (int i = 1; i < n; ++i) {
	// 	if (i > far)
	// 		return false;
	// 	far = max(far, i + nums[i]);
	// 	if (far >= n - 1)
	// 		return true;
	// }

	// return far >= n - 1;

	// return backtracking(0, nums);

	vector<Index> memo(n, Index::UNKNOW);
	memo[n - 1] = Index::GOOD;

	for (int i = n - 2; i >= 0; --i)
	{
		int far = min(i + nums[i], n - 1);
		for (int j = i + 1; j <= far; ++j)
			if (memo[j] == Index::GOOD)
			{
				memo[i] = Index::GOOD;
				break;
			}
	}

	return memo[0] == Index::GOOD;
}

int jump(vector<int>& nums) {
	// int n = nums.size();

	// if (n == 1)
	// 	return 0;

	// vector<int> step(n, n);
	// step[0] = 0;

	// for (int i = 0; i < n; ++i) {
	// 	for (int j = 1; j <= nums[i]; ++j) {
	// 		if (i + j >= n - 1)
	// 			return step[i] + 1;

	// 		if (step[i] + 1 < step[i + j])
	// 			step[i + j] = step[i] + 1;
	// 	}
	// }

	// return step[n - 1];

	int n = nums.size();
	int end = 0, farthest = 0;
	int jumps = 0;
	for (int i = 0; i < n - 1; i++)
	{
		farthest = max(nums[i] + i, farthest);
		if (i == end)
		{
			jumps++;
			end = farthest;
		}
	}

	return jumps;
}

int maxSubArray(vector<int>& nums) {
	int n = nums.size();
	int ans = nums[0];

	int sum = nums[0];

	for (int i = 1; i < n; ++i)
	{
		sum = max(sum + nums[i], nums[i]);
		ans = max(ans, sum);
	}

	return ans;
}

int backtracking(vector<int>& nums, int left, int right) {
	if (left > right)
		return 0;
	if (left == right)
		return nums[left];

	int m1 = max(backtracking(nums, left + 2, right) + nums[left],
				 backtracking(nums, left, right - 2) + nums[right]);
	return max(m1, backtracking(nums, left + 1, right - 1));
}

int simpleRob(vector<int>& nums, int left, int right) {
	if (left > right)
		return 0;

	int n = right - left + 1;
	if (n == 1)
		return nums[left];

	vector<int> dp(n + 1, 0);

	for (int i = 2; i <= n; ++i)
	{ dp[i] = max(dp[i - 1], nums[left + i - 2] + dp[i - 2]); }

	return dp[n];
}

int houseRobII(vector<int>& nums) {
	int n = nums.size();

	if (n == 1)
		return nums[0];

	int left = 0, right = n - 1;

	int m1 = nums[left] + simpleRob(nums, left + 2, right - 1);
	int m2 = nums[right] + simpleRob(nums, left + 1, right - 2);
	int m3 = simpleRob(nums, left + 1, right - 1);

	return max(max(m1, m2), m3);
}

bool backtracking(vector<int>& nums,
				  int currSum,
				  const int tag,
				  const int k,
				  int cnt,
				  int left,
				  unordered_map<int, bool>& memo,
				  int& mask) {
	int n = nums.size();
	if (cnt == k - 1)
		return true;

	if (currSum > tag)
		return false;

	if (memo.count(mask))
		return memo[mask];

	if (currSum == tag)
		return memo[mask] =
				   backtracking(nums, 0, tag, k, cnt + 1, 0, memo, mask);

	if (left >= n)
		return false;

	for (int i = left; i < n; ++i)
	{
		if (((mask >> i) & 1) == 0)
		{
			mask = mask | (1 << i);

			if (backtracking(nums, currSum + nums[i], tag, k, cnt, i + 1, memo,
							 mask))
				return memo[mask] = true;

			mask = (mask ^ (1 << i));
		}
	}

	return false;
}

bool canPartitionKSubsets(vector<int>& nums, int k) {
	int sum = accumulate(nums.begin(), nums.end(), 0);

	if (sum % k != 0)
		return false;

	int tag = sum / k;

	int n = nums.size();
	vector<bool> dp(n, false);

	int mask = 0;
	unordered_map<int, bool> memo;

	sort(nums.begin(), nums.end(), greater<int>());

	return backtracking(nums, 0, tag, k, 0, 0, memo, mask);
}

void backtracking(
	vector<string>& ans, string& str, const int max, int nl, int nr) {
	if (nl == max)
	{
		while (nr < max)
		{
			str.push_back(')');
			nr++;
		}
		ans.push_back(str);
		return;
	}

	str.push_back('(');
	backtracking(ans, str, max, nl + 1, nr);
	str = str.substr(0, nl + nr);

	if (nr < nl)
	{
		str.push_back(')');
		backtracking(ans, str, max, nl, nr + 1);
	}
}

vector<string> generateParenthesis(int n) {
	vector<string> ans;
	string str;
	backtracking(ans, str, 3, 0, 0);
	return ans;
}

int uniquePaths(int m, int n) {
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	dp[1][0] = 1;

	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{ dp[i][j] = dp[i - 1][j] + dp[i][j - 1]; }
	}

	return dp[m][n];
}

int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
	int m = obstacleGrid.size(), n = obstacleGrid[0].size();

	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	dp[1][0] = 1;

	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{
			if (obstacleGrid[i - 1][j - 1] == 1)
				dp[i][j] = 0;
			else
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
		}
	}

	return dp[m][n];
}

// vector<TreeNode*> backtracking(int start, int end) {
// 	vector<TreeNode*> res;

// 	if (start > end) {
// 		res.push_back(nullptr);
// 		return res;
// 	}

// 	for (int i = start; i <= end; ++i) {
// 		vector<TreeNode*> left_sub, right_sub;
// 		left_sub = backtracking(start, i - 1);
// 		right_sub = backtracking(i + 1, end);

// 		for (auto l : left_sub) {
// 			for (auto r : right_sub) {
// 				TreeNode* root = new TreeNode(i);

// 				root->left = l;
// 				root->right = r;

// 				res.push_back(root);
// 			}
// 		}
// 	}

// 	return res;
// }

// vector<TreeNode*> generateTrees(int n) {
// 	return backtracking(1, n);
// }

int backtracking(int start, int end, vector<vector<int>>& dp) {
	if (start >= end)
		return 1;

	if (dp[start][end] != -1)
		return dp[start][end];

	int sum = 0;
	for (int i = start; i <= end; ++i)
	{
		int left_tree = backtracking(start, i - 1, dp);
		int right_tree = backtracking(i + 1, end, dp);
		sum += left_tree * right_tree;
	}
	dp[start][end] = sum;
	return sum;
}

int numTrees(int n) {
	vector<vector<int>> dp(n + 1, vector<int>(n + 1, -1));
	return backtracking(1, n, dp);
}

bool isInterleave(string s1, string s2, string s3) {
	vector<string> dp1;
	vector<string> dp2;
	int start = 0;
	int l1 = 0;
	int l2 = 0;
	while (start < s3.length())
	{
		bool change = false;
		int i = l1;
		for (; i < s1.length();)
		{
			if (s1[i] == s3[start])
			{
				++i;
				++start;
				change = true;
			}
			else if (i != l1)
			{
				dp1.push_back(s1.substr(l1, i - l1));
				break;
			}
			else
			{ break; }
		}
		l1 = i;

		int j = l2;
		for (; j < s2.length();)
		{
			if (s2[j] == s3[start])
			{
				++j;
				++start;
				change = true;
			}
			else if (j != l2)
			{
				dp2.push_back(s2.substr(l2, j - l2));

				break;
			}
			else
			{ break; }
			l2 = j;
		}

		if (!change)
			return false;
	}

	int n = dp1.size() - dp2.size();
	return n >= -1 && n <= 1 && (l1 == s1.size()) && l2 == s2.size();
}

void print_bool(bool val) {
	cout << (val ? "True" : "False") << endl;
}

int lengthOfLIS(vector<int>& nums) {
	int n = nums.size();
	vector<int> dp(n, 1);

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			if (nums[i] > nums[j])
			{ dp[i] = max(dp[i], dp[j] + 1); }
		}
	}

	int ans = 0;
	for (int i = 0; i < n; ++i)
		ans = max(ans, dp[i]);

	return ans;
}

int fibHelp(int n, map<int, int>& dp) {
	if (!dp.count(n))
		dp[n] = fibHelp(n - 1, dp) + fibHelp(n - 2, dp);
	return dp[n];
}

int fib(int n) {
	map<int, int> dp;
	dp[0] = 0;
	dp[1] = 1;

	return fibHelp(n, dp);
}

int binarySearch(vector<int>& sub, int num) {
	int left = 0;
	int right = sub.size() - 1;
	int mid = (left + right) / 2;

	while (left < right)
	{
		mid = (left + right) / 2;
		if (sub[mid] == num)
			return mid;

		if (sub[mid] < num)
			left = mid + 1;
		else
			right = mid;
	}

	return left;
}

int lengthOfLISII(vector<int>& nums) {
	vector<int> sub;
	sub.push_back(nums[0]);

	for (int i = 1; i < nums.size(); ++i)
	{
		int num = nums[i];
		if (num > sub[sub.size() - 1])
			sub.push_back(num);
		else
		{
			int j = binarySearch(sub, num);
			sub[j] = num;
		}
	}

	return sub.size();
}

int maxEnvelopes(vector<vector<int>>& envelopes) {
	int n = envelopes.size();
	sort(envelopes.begin(), envelopes.end(),
		 [](vector<int>& a, vector<int>& b) {
			 return a[0] < b[0] || (a[0] == b[0] && a[1] > b[1]);
		 });

	vector<int> LIS;
	for (auto& envelope : envelopes)
	{ LIS.push_back(envelope[1]); }

	return lengthOfLISII(LIS);
}

int minDistance(string word1, string word2) {
	int m = word1.length(), n = word2.length();

	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	for (int i = 1; i <= m; ++i)
		dp[i][0] = i;
	for (int j = 1; j <= n; ++j)
		dp[0][j] = j;

	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{
			if (word1[i - 1] == word2[j - 1])
				dp[i][j] = dp[i - 1][j - 1];
			else
				dp[i][j] = min(dp[i - 1][j] + 1,
							   min(dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1));
		}
	}

	return dp[m][n];
}

int maxProfit(vector<int>& prices, int fee) {
	int m = prices.size();

	vector<vector<int>> dp(m + 1, vector<int>(2, 0));
	dp[1][1] = -prices[0];

	for (int i = 2; i <= m; ++i)
	{
		dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1] - fee);
		dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1]);
	}

	return max(dp[m][0], dp[m][1]);
}

int calculateMinimumHP(vector<vector<int>>& dungeon) {
	int m = dungeon.size();
	int n = dungeon[0].size();

	vector<vector<int>> dp(m + 1, vector<int>(n + 1, INT_MAX));
	dp[m][n - 1] = dp[m - 1][n] = 1;

	for (int i = m - 1; i >= 0; --i)
	{
		for (int j = n - 1; j >= 0; --j)
		{
			dp[i][j] = min(dp[i][j + 1], dp[i + 1][j]) - dungeon[i][j];
			if (dp[i][j] <= 0)
				dp[i][j] = 1;
		}
	}

	return dp[0][0];
}

int backtracking(vector<int>& nums,
				 vector<int>& prices,
				 map<int, int>& dp,
				 int i) {
	if (i >= nums.size())
		return 0;

	if (dp.count(i))
		return dp[i];

	int num = nums[i];
	int price = prices[i];

	int index = i;

	int v1 = backtracking(nums, prices, dp, ++i);

	if (i < nums.size() && nums[i] == num + 1)
		++i;

	int v2 = num * price + backtracking(nums, prices, dp, i);
	dp[index] = max(v1, v2);
	return dp[index];
}

int deleteAndEarn(vector<int> nums) {
	map<int, int> data;
	for (auto num : nums)
	{ data[num]++; }

	vector<int> allNums;
	vector<int> numPrices;
	for (auto& val : data)
	{
		allNums.push_back(val.first);
		numPrices.push_back(val.second);
	}

	map<int, int> dp;
	return backtracking(allNums, numPrices, dp, 0);
}

int searchII(vector<int>& nums, int target) {
	int left = 0, right = nums.size() - 1;
	int mid = (left + right) / 2;
	while (left < right)
	{
		mid = (left + right) / 2;
		if (nums[mid] == target)
			return mid;
		else if (nums[mid] < target)
			left = mid + 1;
		else
			right = mid - 1;
	}

	return (nums[left] == target ? left : -1);
}

int searchInsert(vector<int>& nums, int target) {
	int left = 0, right = nums.size() - 1;

	while (left < right)
	{
		int mid = (left + right) / 2;
		if (nums[mid] == target)
			return mid;
		else if (nums[mid] < target)
			left = mid + 1;
		else
			right = mid - 1;
	}

	if (nums[left] >= target)
		return left;
	else
		return left + 1;
}

bool isSubsequence(string s, string t) {
	int i = 0, j = 0;

	while (i < s.length() && j < t.length())
	{
		if (s[i] == t[j])
			++i;
		++j;
	}

	return i == s.length();
}

int trailingZeroes(int n) {
	int ans = 0;
	for (int i = n; i / 5 > 0; i = i / 5)
	{ ans += i / 5; }
	return ans;
}

long trailingZeroes(long n) {
	long ans = 0;
	for (long i = n; i / 5 > 0; i = i / 5)
	{ ans += i / 5; }
	return ans;
}

long left_bound(int target) {
	long lo = 0, hi = LONG_MAX;
	while (lo < hi)
	{
		long mid = lo + (hi - lo) / 2;
		if (trailingZeroes(mid) < target)
			lo = mid + 1;
		else
			hi = mid;
	}

	return lo;
}

long right_bound(int target) {
	long lo = 0, hi = LONG_MAX;
	while (lo < hi)
	{
		long mid = lo + (hi - lo) / 2;
		if (trailingZeroes(mid) > target)
			hi = mid - 1;
		else
			lo = mid;
	}

	return hi;
}

int preimageSizeFZF(int k) {
	return right_bound(k) - left_bound(k);
}

int minEatingSpeed(vector<int>& piles, int h) {
	int left = 0, right = 0, sum = 0;

	for (auto pile : piles)
	{
		right = max(right, pile);
		sum += pile;
	}

	left = sum / h;

	return 0;
}

bool hasCycle(ListNode* head) {
	if (!head)
		return false;

	ListNode* slow = head;
	ListNode* fast = slow->next;

	int step = 1;

	while (fast != nullptr && slow != nullptr)
	{
		if (fast == slow || fast->next == slow)
			return true;
		if (step % 2 == 0)
			slow = slow->next;
		fast = fast->next;
		step++;
	}

	return false;
}

int shipWithinDays(vector<int> weights, int days) {
	int sum = accumulate(weights.begin(), weights.end(), 0);
	int avg = sum / days;

	int right = 2 * avg;
	int left = weights[0];
	for (auto weight : weights)
		left = max(left, weight);

	while (left < right)
	{
		int mid = left + (right - left) / 2;

		int cnt = 0;
		int tmp = 0;
		for (int i = 0; i < weights.size(); ++i)
		{
			int weight = weights[i];
			tmp += weight;
			if (tmp > mid)
			{
				cnt++;
				tmp = weight;
			}
			if (cnt > days)
			{
				left = mid + 1;
				break;
			}
			else if (i == weights.size() - 1)
			{
				cnt++;
				if (cnt > days)
					left = mid + 1;
				else
					right = mid;
			}
		}
	}

	return left;
}

// int lengthOfLongestSubstring(string s) {
// 	vector<int> chars(128, -1);
// 	int left = 0, right = 0, ans = 0;

// 	for (; right < s.length(); ++right)
// 	{
// 		int offset = s[right];

// 		if (chars[offset] >= 0 && chars[offset] >= left)
// 		{
// 			ans = max(ans, right - left);
// 			left = chars[offset] + 1;
// 		}

// 		chars[offset] = right;
// 	}

// 	return max(ans, right - left);
// }

TreeNode* buildTreeHelp(vector<int>& preorder,
						vector<int>& inorder,
						int l1,
						int r1,
						int l2,
						int r2) {
	if (l1 > r1 || l2 > r2)
		return nullptr;
	TreeNode* node = new TreeNode(preorder[l1]);
	if (l1 != r1)
	{
		int i = l2;
		for (; i <= r2; ++i)
		{
			if (preorder[l1] == inorder[i])
				break;
		}

		int n = i - l2;

		node->left =
			buildTreeHelp(preorder, inorder, l1 + 1, l1 + n, l2, i - 1);
		node->right =
			buildTreeHelp(preorder, inorder, l1 + n + 1, r1, i + 1, r2);
	}
	return node;
}

TreeNode* buildTree(vector<int> preorder, vector<int> inorder) {
	int n = preorder.size();
	return buildTreeHelp(preorder, inorder, 0, n - 1, 0, n - 1);
}

ListNode* deleteDuplicates(ListNode* head) {
	ListNode* pprev = nullptr;
	ListNode* prev = nullptr;
	ListNode* cur = head;
	int cnt = 0;

	while (cur != nullptr)
	{
		if (prev != nullptr && prev->val == cur->val)
		{
			prev->next = cur->next;
			delete cur;
			cur = prev->next;
			cnt++;
		}
		else
		{
			if (cnt > 0)
			{
				delete prev;
				prev = cur;
				cur = cur->next;
				cnt = 0;
				if (pprev == nullptr)
					head = prev;
				else
					pprev->next = prev;
			}
			else
			{
				pprev = prev;
				prev = cur;
				cur = cur->next;
			}
		}
	}

	if (cnt > 0)
	{
		delete prev;
		prev = cur;
		cnt = 0;
		if (pprev == nullptr)
			head = prev;
		else
			pprev->next = prev;
	}

	return head;
}

ListNode* MergeTwoLists(ListNode* list1, ListNode* list2) {
	// write code here
	ListNode* l1 = list1;
	ListNode* l2 = list2;
	ListNode* head = nullptr;
	ListNode* prev = nullptr;

	while (l1 && l2)
	{
		ListNode* cur = l1->val > l2->val ? l1 : l2;
		if (head == nullptr)
		{ head = prev = cur; }
		else
		{
			prev->next = cur;
			prev = cur;
		}

		l1 = l1 == cur ? l1->next : l1;
		l2 = l2 == cur ? l2->next : l2;
	}

	if (l1 && prev)
		prev->next = l1;
	if (l2 && prev)
		prev->next = l2;

	return head;
}

int numKLenSubstrRepeats(string s, int k) {
	map<char, int> items;
	int cnt = 0;
	for (int i = 0; i < s.size(); ++i)
	{
		if (items.count(s[i]) == 0)
			items[s[i]] = 1;
		else
			items[s[i]]++;

		if (i >= k - 1)
		{
			if (items.size() < k)
				cnt++;

			if (items[s[i - k + 1]] == 1)
				items.erase(s[i - k + 1]);
			else
				items[s[i - k + 1]]--;
		}
	}
	return cnt;
}

// -1 0 1 0 -1
void minDirect(vector<vector<int>>& heights,
			   vector<vector<bool>>& visited,
			   int& _x,
			   int& _y) {
	int effort = INT_MAX;

	int dir = -1;
	for (int i = 0; i < 4; ++i)
	{
		int x = _x + direction[i];
		int y = _y + direction[i + 1];

		if (x >= 0 && x <= heights.size() && y >= 0 && y <= heights[0].size() &&
			!visited[x][y])
		{ int cur_effort = max(0, heights[x][y] - heights[_x][_y]); }
	}
}

// vector<int> diffWaysToCompute(string expression) {
// 	vector<int> ways;
// 	for (int i = 0; i < expression.size(); ++i)
// 	{
// 		char c = expression[i];
// 		if (c == '+' || c == '-' || c == '*')
// 		{
// 			vector<int> left = diffWaysToCompute(expression.substr(0, i));
// 			vector<int> right = diffWaysToCompute(expression.substr(i + 1));

// 			for (const int& l : left)
// 			{
// 				for (const int& r : right)
// 				{
// 					switch (c)
// 					{
// 						case '+': ways.push_back(l + r); break;
// 						case '-': ways.push_back(l - r); break;
// 						case '*': ways.push_back(l * r); break;
// 					}
// 				}
// 			}
// 		}
// 	}

// 	if (ways.empty())
// 		ways.push_back(stoi(expression));
// 	return ways;
// }

vector<int> diffWaysToCompute(string expression) {
	vector<int> data;
	vector<int> ops;

	int num = 0;
	char op = ' ';
	istringstream ss(expression + "+");
	while (ss >> num && ss >> op)
	{
		data.push_back(num);
		ops.push_back(op);
	}

	int n = data.size();
	vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>()));

	for (int i = 0; i < n; ++i)
	{
		for (int j = i; j >= 0; --j)
		{
			if (i == j)
			{ dp[j][i].push_back(data[i]); }
			else
			{
				for (int k = j; k < i; ++k)
				{
					for (auto left : dp[j][k])
					{
						for (auto right : dp[k + 1][i])
						{
							int val = 0;
							switch (ops[k])
							{
								case '+': val = left + right; break;
								case '-': val = left - right; break;
								case '*': val = left * right; break;
							}
							dp[j][i].push_back(val);
						}
					}
				}
			}
		}
	}

	return dp[0][n - 1];
}

int main() {
	// vector<vector<int>> envelopes{
	// 	{46, 89}, {50, 53}, {52, 68}, {72, 45}, {77, 81}};

	// cout << maxEnvelopes(envelopes) << endl;

	// vector<int> nums{1, 1, 1};
	// cout << lengthOfLISII(nums) << endl;

	// cout << minDistance("horse", "ros") << endl;

	// cout << deleteAndEarn(vector<int>{2, 2, 3, 3, 3, 4}) << endl;

	// int value = 1;
	// int i = 1;
	// for (;; ++i)
	// {
	// 	if (value >= INT_MAX / i)
	// 		break;
	// 	value = value * i;
	// 	cout << i << " " << value << endl;
	// }

	// cout << "INT_MAX " << INT_MAX << endl;

	// cout << shipWithinDays(vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5)
	// 	 << endl;

	// TreeNode* root =
	// 	buildTree(vector<int>{3, 9, 20, 15, 7}, vector<int>{9, 3, 15, 20, 7});

	// recoverTree(root);

	vector<int> nums{3, 9, 20, 15, 7};
	Heap maxHeap(nums);
	maxHeap.print();

	maxHeap.Pop();
	maxHeap.print();

	maxHeap.Push(8);
	maxHeap.print();

	maxHeap.Push(4);
	maxHeap.print();

	maxHeap.Push(10);
	maxHeap.print();

	maxHeap.Push(6);
	maxHeap.print();

	maxHeap.Push(16);
	maxHeap.print();

	// vector<ListNode*> nodes;
	// for (auto num : vector<int>{1, 1})
	// { nodes.push_back(new ListNode(num)); }

	// for (int i = 0; i < nodes.size() - 1; ++i)
	// 	nodes[i]->next = nodes[i + 1];

	// ListNode* head = deleteDuplicates(nodes[0]);

	// ListNode* list1 = new ListNode(11);
	// ListNode* l1 = list1;
	// l1->next = new ListNode(5);
	// l1 = l1->next;
	// l1->next = new ListNode(1);

	// ListNode* list2 = new ListNode(6);
	// ListNode* l2 = list2;
	// l2->next = new ListNode(3);
	// l2 = l2->next;
	// l2->next = new ListNode(2);

	// ListNode* head = MergeTwoLists(list1, list2);

	cout << numKLenSubstrRepeats("yokagames", 3) << endl;

	return 0;
}