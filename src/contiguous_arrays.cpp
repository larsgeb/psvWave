
#include <initializer_list>
#include <assert.h>
#include <vector>
#include <iostream>

int linear_IDX(int pos1, int shape1)
{
  // assert(pos1 < shape1);
  // assert(pos1 >= 0);
  return pos1;
}

int linear_IDX(int pos1, int pos2, int shape1, int shape2)
{
  // assert(pos1 < shape1);
  // assert(pos1 >= 0);
  // assert(pos2 < shape2);
  // assert(pos2 >= 0);
  return pos1 * shape2 + pos2;
}

int linear_IDX(int pos1, int pos2, int pos3, int shape1, int shape2, int shape3)
{
  // assert(pos1 < shape1);
  // assert(pos1 >= 0);
  // assert(pos2 < shape2);
  // assert(pos2 >= 0);
  // assert(pos3 < shape3);
  // assert(pos3 >= 0);
  return (pos1 * shape2 + pos2) * shape3 + pos3;
}

int linear_IDX(int pos1, int pos2, int pos3, int pos4, int shape1, int shape2, int shape3, int shape4)
{
  // assert(pos1 < shape1);
  // assert(pos1 >= 0);
  // assert(pos2 < shape2);
  // assert(pos2 >= 0);
  // assert(pos3 < shape3);
  // assert(pos3 >= 0);
  // assert(pos4 < shape4);
  // assert(pos4 >= 0);
  return ((pos1 * shape2 + pos2) * shape3 + pos3) * shape4 + pos4;
}
