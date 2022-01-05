#include <vector>
#include <set>
#include <map>
#include <list>
#include <queue>
#include <memory>
#include <algorithm>
#include <limits.h>
#include <cstddef>
#include <algorithm>
#include <typeinfo>
#include <iostream>
#include <stack>
#include "math.h"

using namespace std;

// 455
int findContentChildren(vector<int>& children, vector<int>& cookies)
{
	sort(children.begin(), children.end());
	sort(cookies.begin(), cookies.end());

	int child = 0, cookie = 0;
	while (child < children.size() && cookie < cookies.size()) {
		if (children[child] <= cookies[cookie]) child++;
		cookie++;
	}

	return child;
}

// 135
int candy(vector<int>& ratings)
{
	vector<int> candies(ratings.size(), 1);
	return 0;
}

// 605
bool canPlaceFlowers(vector<int>& flowerbed, int n)
{
	size_t size = flowerbed.size();
	int indicator = 0;
	int max = 0;
	for (size_t i = 0; i < size; ++i) {
		if (flowerbed[i] == 1) {
			if (indicator != 0) { max--; }
			indicator = 1;
		}
		else {
			if (indicator == 0) {
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
int findMinArrowShots(vector<vector<int>>& points)
{
	sort(points.begin(), points.end(),
		 [](vector<int>& a, vector<int>& b) { return a[0] < b[0]; });

	int arrows = 1;
	int indicator = points[0][1];
	int left = points[0][0];
	int right = points[0][1];
	for (auto point : points) {
		if (point[0] <= right) {
			left = max(left, point[0]);
			right = min(right, point[1]);
		}
		else {
			arrows++;
			left = point[0];
			right = point[1];
		}
	}
	return arrows;
}

// 763
vector<int> partitionLabels(string s)
{
	map<char, int> char_info;
	for (int i = 0; i < s.size(); ++i) {
		if (char_info.find(s[i]) == char_info.end())
			char_info.insert(pair<char, int>(s[i], i));
		else
			char_info[s[i]] = i;
	}

	vector<vector<int>> regions;
	for (int i = 0; i < s.size(); ++i) {
		bool inside = false;
		for (vector<int>& reg : regions) {
			if (i <= reg[1] && i >= reg[0]) {
				inside = true;
				reg[0] = min(reg[0], i);
				reg[1] = max(reg[1], char_info[s[i]]);
				break;
			}
		}

		if (inside == false) regions.push_back({i, char_info[s[i]]});
	}

	vector<int> len;
	for (auto reg : regions) {
		len.push_back(reg[1] - reg[0] + 1);
	}

	return len;
}

// 122
int maxProfit(vector<int>& prices)
{
	int profit = 0;
	int size = prices.size();

	int prev = INT_MAX;
	for (int i = 0; i < size; ++i) {
		if (prices[i] > prev) profit += (prices[i] - prev);
		prev = prices[i];
	}

	return profit;
}

// 406
vector<vector<int>> reconstructQueue(vector<vector<int>>& people)
{
	map<int, vector<int>, greater<int>> group;
	for (auto value : people) {
		if (group.find(value[0]) == group.end())
			group[value[0]] = {value[1]};
		else
			group[value[0]].push_back(value[1]);
	}

	for (auto it = group.begin(); it != group.end(); ++it) {
		sort((*it).second.begin(), (*it).second.end());
	}

	vector<vector<int>> res;
	for (auto it = group.begin(); it != group.end(); ++it) {
		for (auto index : (*it).second) {
			vector<int> pair{(*it).first, index};
			res.insert(res.begin() + index, pair);
		}
	}

	return res;
}

struct ListNode
{
	int val;
	ListNode* next;
	ListNode(int x) : val(x), next(NULL) {}
};

// 142
ListNode* detectCycle(ListNode* head)
{
	ListNode *fast = head, *slow = head;
	do {
		if (!fast || !fast->next) return nullptr;

		fast = fast->next->next;
		slow = slow->next;
	} while (fast != slow);

	fast = head;
	while (fast != slow) {
		fast = fast->next;
		slow = slow->next;
	}

	return fast;
}

// 88
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n)
{
	int pos = m-- + n-- - 1;
	while (m >= 0 && n >= 0) {
		nums1[pos--] = nums1[m] > nums2[n] ? nums1[m--] : nums2[n--];
	}
	while (n >= 0) {
		nums1[pos--] = nums2[n--];
	}
}

// 633
bool judgeSquareSum(int c)
{
	int min = 0, max = int(sqrt(c));
	for (; min <= max;) {
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
bool validSubStr(string& s, int l, int r)
{
	while (l < r) {
		if (s[l] != s[r]) return false;
		++l;
		--r;
	}
	return true;
}

bool validPalindrome(string s)
{
	// vector<char> chars(s.begin(), s.end());
	int size = s.size();
	int l = 0, r = size - 1;

	while (l < r) {
		if (s[l] != s[r]) {
			return (validSubStr(s, l + 1, r) || validSubStr(s, l, r - 1));
		}

		++l;
		--r;
	}

	return true;
}

// 524
string findLongestWord(string s, vector<string>& dictionary)
{
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
	for (auto& word : dictionary) {
		// no need to compare
		if (word.size() > s.size() || max_len > word.size()) {
			++index;
			continue;
		}

		int cnt = 0;
		for (int i = 0, j = 0; i < s.size() && j < word.size();) {
			if (s[i] != word[j]) { ++i; }
			else {
				++i;
				++j;
				cnt++;
			}
		}

		if (cnt == word.size()) {
			if (cnt > max_len || (cnt == max_len && word < dictionary[max_i])) {
				max_len = cnt;
				max_i = index;
			}
		}

		++index;
	}

	return max_i >= 0 ? dictionary[max_i] : "";
}

// 340
int lengthOfLongestSubstringKDistinct(string s, int k)
{
	if (k == 0) return 0;

	int size = s.size();

	if (k >= size) return size;

	map<char, int> c_num;
	int max = 0;
	int l = 0, r = 0;
	for (; l < size && r < size;) {
		auto itor = c_num.find(s[r]);
		if (k == 0 && itor == c_num.end()) {
			max = (r - l) > max ? (r - l) : max;

			for (; l < r; ++l) {
				c_num[s[l]]--;
				if (c_num[s[l]] == 0) {
					c_num.erase(s[l]);
					l++;
					break;
				}
			}
			k++;
			continue;
		}

		if (itor == c_num.end()) {
			--k;
			c_num[s[r]] = 1;
		}
		else {
			c_num[s[r]]++;
		}

		r++;
	}

	max = (r - l) > max ? (r - l) : max;
	return max;
}

// 3
int lengthOfLongestSubstring(string s)
{
	int size = s.size();

	set<char> chars;
	int max = 0;

	int l = 0, r = 0;
	for (; r < size; ++r) {
		if (chars.find(s[r]) == chars.end()) { chars.insert(s[r]); }
		else {
			max = (r - l) > max ? (r - l) : max;
			for (; l < r;) {
				if (s[l] == s[r]) {
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
int lengthOfLongestSubstringTwoDistinct(string s)
{
	/*
	int n = s.size();

	if (n < 3)
		return n;

	map<char, int> chars;  //<char, index>
	int max = 0;

	int l = 0, r = 0;
	while (r < n) {
		chars[s[r]] = r;
		if (chars.size() == 3) {
			int del_i = INT_MAX;
			auto itor = chars.begin();
			for (; itor != chars.end(); ++itor) {
				del_i = min(del_i, itor->second);
			}
			max = (r - l) > max ? (r - l) : max;
			l = del_i + 1;
			chars.erase(s[del_i]);
		}
		r++;
	}

	return (r - l) > max ? (r - l) : max;
	*/
	int n = s.size();
	vector<char> chars(128);
	int k = 2;

	int l = 0, r = 0;
	int max = 0;

	for (; r < n;) {
		if (chars[s[r]] == 0) {
			if (k == 0) {
				max = (r - l > max) ? (r - l) : max;
				while (l < r) {
					chars[s[l]]--;
					if (chars[s[l]] == 0) {
						++l;
						chars[s[r]] = 1;
						break;
					}
					++l;
				}
			}
			else {
				chars[s[r]] = 1;
				--k;
			}
		}
		else {
			chars[s[r]]++;
		}
		++r;
	}

	return (max = (r - l > max) ? (r - l) : max);
}

// 69
int mySqrt(int x)
{
	if (x == 0) return 0;
	int l = 1, r = x;
	int mid, sqrt;

	while (l <= r) {
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
bool isPerfectSquare(int num)
{
	if (num == 1) return true;

	int l = 1, r = num;

	int mid, sqrt;

	while (l <= r) {
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

// 154
int findMin(vector<int>& nums)
{
	int n = nums.size();

	int l = 0, r = n - 1;
	int min = nums[0];
	while (l <= r) {
		int mid = l + (r - l) / 2;

		if (nums[mid] == nums[l]) {
			min = std::min(min, nums[l]);
			l++;
		}
		else if (nums[mid] <= nums[r])	// right side is in order and mid is
										// minimum in right side
		{
			min = std::min(min, nums[mid]);
			r = mid - 1;
		}
		else {
			min = std::min(min, nums[mid]);
			l = mid + 1;
		}
	}

	return min;
}

// 215

int quickSelection(vector<int>& nums, int l, int r)
{
	int i = l + 1, j = r;

	while (true) {
		while (i < r && nums[i] <= nums[l])
			++i;

		while (l < j && nums[j] >= nums[l])
			--j;

		if (i >= j) break;

		swap(nums[i], nums[j]);
	}

	swap(nums[l], nums[j]);
	return j;
}

int findKthLargest(vector<int>& nums, int k)
{
	int l = 0, r = nums.size() - 1, target = nums.size() - k;

	while (l < r) {
		int mid = quickSelection(nums, l, r);
		if (mid == target) return nums[mid];

		if (mid < target)
			l = mid + 1;
		else
			r = mid - 1;
	}

	return nums[l];
}

// 324
void wiggleSort(vector<int>& nums) {}

class KthLargest {
private:
	priority_queue<int, vector<int>, greater<int>> topNums;
	int max_k;

public:
	KthLargest(int k, vector<int>& nums)
	{
		max_k = k;
		for (auto& num : nums) {
			topNums.push(num);
		}

		while (topNums.size() > k) {
			topNums.pop();
		}
	}

	int add(int val)
	{
		topNums.push(val);

		while (topNums.size() > max_k) {
			topNums.pop();
		}

		return topNums.top();
	}
};

// 547
int findCircleNum_1(vector<vector<int>>& isConnected)
{
	int n = isConnected.size();

	int stateNum = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = i; j < n; ++j) {
			if (isConnected[i][j]) {
				isConnected[i][j] = 0;

				stateNum++;

				stack<int> cities;
				cities.push(j);

				while (!cities.empty()) {
					int c = cities.top();
					cities.pop();

					for (int k = 0; k < n; ++k) {
						if (isConnected[c][k]) {
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
void dfs(vector<vector<int>>& isConnected, vector<bool>& visited, int i)
{
	visited[i] = true;

	for (int k = 0; k < isConnected.size(); ++k) {
		if (isConnected[i][k] && !visited[k]) dfs(isConnected, visited, k);
	}
}

int findCircleNum(vector<vector<int>>& isConnected)
{
	int n = isConnected.size(), count = 0;

	vector<bool> visited(n, false);

	for (int i = 0; i < n; ++i) {
		if (!visited[i]) {
			dfs(isConnected, visited, i);
			++count;
		}
	}

	return count;
}

vector<int> direction{-1, 0, 1, 0, -1};

// 417
void dfs(const vector<vector<int>>& matrix,
		 vector<vector<bool>>& can_reach,
		 int r,
		 int c)
{
	if (can_reach[r][c]) return;

	can_reach[r][c] = true;

	int x, y;
	for (int i = 0; i < 4; ++i) {
		x = r + direction[i], y = c + direction[i + 1];

		if (x >= 0 && x < matrix.size() && y >= 0 && y < matrix[0].size() &&
			matrix[r][c] <= matrix[x][y])
			dfs(matrix, can_reach, x, y);
	}
}

vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix)
{
	if (matrix.empty() || matrix[0].empty()) return {};

	vector<vector<int>> res;

	int m = matrix.size(), n = matrix[0].size();

	vector<vector<bool>> can_reach_p(m, vector<bool>(n, false));
	vector<vector<bool>> can_reach_a(m, vector<bool>(n, false));

	for (int i = 0; i < m; ++i) {
		dfs(matrix, can_reach_p, i, 0);
		dfs(matrix, can_reach_a, i, n - 1);
	}

	for (int i = 0; i < n; ++i) {
		dfs(matrix, can_reach_p, 0, i);
		dfs(matrix, can_reach_a, m - 1, i);
	}

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (can_reach_a[i][j] && can_reach_p[i][j])
				res.push_back(vector<int>{i, j});
		}
	}

	return res;
}

// 47
void backtracking(vector<int>& nums, int level, vector<vector<int>>& ans)
{
	if (level == nums.size() - 1) {
		ans.push_back(nums);
		return;
	}

	for (int i = level; i < nums.size(); i++) {
		swap(nums[i], nums[level]);
		backtracking(nums, level + 1, ans);
		swap(nums[i], nums[level]);
	}
}
vector<vector<int>> permute(vector<int>& nums)
{
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
				  int pos)
{
	if (level == k) {
		res.push_back(tmp);
		return;
	}

	for (int i = pos; i < n; ++i) {
		tmp[level++] = i + 1;
		backtracking(res, tmp, n, k, level, i + 1);
		level--;
	}
}

vector<vector<int>> combine(int n, int k)
{
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
				  bool& find)
{
	if (visited[r][c] || board[r][c] != word[i]) return;

	visited[r][c] = true;

	if (i + 1 == word.size()) {
		find = true;
		return;
	}

	for (int k = 0; k < 4; ++k) {
		int x = r + direction[k], y = c + direction[k + 1];

		if (x < 0 || x >= board.size() || y < 0 || y >= board[0].size())
			continue;

		i++;

		backtracking(board, word, i, visited, x, y, find);
		if (find) return;
		i--;
	}

	visited[r][c] = false;
}

bool exist(vector<vector<char>>& board, string word)
{
	if (board.empty() || board[0].empty()) return false;
	vector<vector<bool>> visited(board.size(),
								 vector<bool>(board[0].size(), false));
	bool find = false;

	for (int x = 0; x < board.size(); ++x) {
		for (int y = 0; y < board[0].size(); ++y) {
			backtracking(board, word, 0, visited, x, y, find);
			if (find) return true;
		}
	}

	return find;
}

// 130
void backtracking(vector<vector<char>>& board,
				  vector<vector<bool>>& visited,
				  vector<pair<int, int>>& points,
				  char target,
				  int r,
				  int c)
{
	if (visited[r][c] || board[r][c] != target) return;

	visited[r][c] = true;
	points.push_back(pair<int, int>{r, c});

	for (int i = 0; i < 4; i++) {
		int x = r + direction[i], y = c + direction[i + 1];

		if (x >= 0 && y >= 0 && x < board.size() && y < board[0].size()) {
			backtracking(board, visited, points, target, x, y);
		}
	}
}

void solve(vector<vector<char>>& board)
{
	int m = board.size(), n = board[0].size();
	vector<vector<bool>> visited(m, vector<bool>(n, false));

	vector<pair<int, int>> points;

	for (int i = 0; i < m; ++i) {
		backtracking(board, visited, points, 'O', i, 0);
		backtracking(board, visited, points, 'O', i, n - 1);
	}

	for (int i = 0; i < n; ++i) {
		backtracking(board, visited, points, 'O', 0, i);
		backtracking(board, visited, points, 'O', m - 1, i);
	}

	points.clear();

	for (int i = 1; i < m - 1; i++) {
		for (int j = 1; j < n - 1; j++) {
			backtracking(board, visited, points, 'O', i, j);
		}
	}

	for (const auto& point : points) {
		board[point.first][point.second] = 'X';
	}
}

// 40
void backtracking(map<int, int>& numCount,
				  map<int, int>& numUsed,
				  vector<vector<int>>& res,
				  map<int, int>::iterator itor,
				  int target,
				  int level)
{
	if (itor->first > target) return;

	if (itor->first == target) {
		vector<int> combine;
		for (auto it : numUsed) {
			if (it.second > 0) {
				combine.insert(combine.end(), it.second, it.first);
			}
		}
		combine.push_back(itor->first);
		res.push_back(combine);

		return;
	}

	int i = level == 0 ? 1 : 0;
	for (; i <= itor->second; ++i) {
		if (itor->first * i > target) break;

		numUsed[itor->first] = i;
		int tag = target - i * itor->first;

		if (tag == 0) {
			vector<int> combine;
			for (auto it : numUsed) {
				if (it.second > 0) {
					combine.insert(combine.end(), it.second, it.first);
				}
			}

			res.push_back(combine);
			break;
		}

		itor++;
		if (itor == numCount.end()) { continue; }
		backtracking(numCount, numUsed, res, itor, tag, ++level);

		itor--;
	}

	numUsed[itor->first] = 0;
}

vector<vector<int>> combinationSum2(vector<int>& candidates, int target)
{
	vector<vector<int>> res;

	map<int, int> numCount;
	map<int, int> numUsed;
	for (auto val : candidates) {
		if (numCount.find(val) != numCount.end()) { numCount[val]++; }
		else {
			numCount[val] = 1;
			numUsed[val] = 0;
		}
	}

	for (auto itor = numCount.begin(); itor != numCount.end(); itor++) {
		backtracking(numCount, numUsed, res, itor, target, 0);
	}

	return res;
}
