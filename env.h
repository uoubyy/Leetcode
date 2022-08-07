#ifndef _ENV_H
#define _ENV_H

#include <array>
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

#include <math.h>
#include <numeric>
#include <unordered_map>
#include <stack>
#include <unordered_set>

#include <thread>
#include <mutex>

#include <time.h>
#include <utility>
#include <bitset>

#include <nlohmann/json.hpp>

using namespace std;

void display_vector(const vector<int>& nums) {
	for (int i = 0; i < nums.size(); ++i)
	{ cout << nums[i] << " "; }
	cout << endl;
}

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

vector<int> direction{-1, 0, 1, 0, -1};

// Definition for a binary tree node.
struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode() : val(0), left(nullptr), right(nullptr) {}
	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
	TreeNode(int x, TreeNode* left, TreeNode* right)
		: val(x), left(left), right(right) {}
};

TreeNode* BuildBinarySearchTree(vector<int>& nums) {
	int n = nums.size();
	if (n <= 0)
		return nullptr;

	TreeNode* root = new TreeNode(nums[0]);
	queue<TreeNode*> nodes;
	nodes.push(root);

	int i = 1;
	while (!nodes.empty())
	{
		TreeNode* curr = nodes.front();
		nodes.pop();

		TreeNode* left = i < n ? new TreeNode(nums[i]) : nullptr;
		TreeNode* right = i + 1 < n ? new TreeNode(nums[i + 1]) : nullptr;

		i += 2;
		curr->left = left;
		curr->right = right;

		if (left != nullptr)
			nodes.push(left);
		if (right != nullptr)
			nodes.push(right);
	}

	return root;
}

vector<vector<int>> muliMatrix(const vector<vector<int>>& mat1,
							   const vector<vector<int>>& mat2) {
	int m1 = mat1.size(), n1 = mat1[0].size(), m2 = mat2.size(),
		n2 = mat2[0].size();

	assert(n1 == m2);

	vector<vector<int>> res(m1, vector<int>(n2, 0));
	for (int i = 0; i < m1; ++i)
	{
		for (int j = 0; j < n2; ++j)
		{
			for (int k = 0; k < m2; ++k)
				res[i][j] += mat1[i][k] * mat2[k][j];
		}
	}

	return res;
}

// 10^75 = 10^(64 + 8 + 2 + 1)
// 75 = 1001011

vector<vector<int>> matrixPower(const vector<vector<int>>& mat, int k) {
	int m = mat.size(), n = mat[0].size();
	assert(m == n);

	vector<vector<int>> res(m, vector<int>(m, 0));
	for (int i = 0; i < m; ++i)
		res[i][i] = 1;

	vector<vector<int>> tmp(mat);

	for (; k != 0; k >>= 1)
	{
		if ((k & 1) != 0)
			res = muliMatrix(res, tmp);
		tmp = muliMatrix(tmp, tmp);
	}

	return res;
}

namespace KMP
{
vector<int> getNextArray(const string& str) {
	int n = str.length();
	if (n == 1)
		return {-1};

	vector<int> next(n, 0);
	next[0] = -1;
	next[1] = 0;

	int pos = 2;
	int cn = 0;
	while (pos < n)
	{
		if (str[pos - 1] == str[cn])
			next[pos++] = ++cn;
		else if (cn > 0)
			cn = next[cn];
		else
			next[pos++] = 0;
	}
	return next;
}

int getIndexOf(const string& str, const string& match) {
	int m = str.length(), n = match.length();

	if (m <= 0 || n <= 0 || m < n)
		return -1;

	vector<int> next = getNextArray(match);

	int si = 0, mi = 0;
	while (si < m && mi < n)
	{
		if (str[si] == match[mi])
		{
			si++;
			mi++;
		}
		else if (next[mi] == -1)
		{ si++; }
		else
		{ mi = next[mi]; }
	}

	return mi == n ? si - mi : -1;
}

}  // namespace KMP

#endif