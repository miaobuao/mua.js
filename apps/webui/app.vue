<script setup lang="ts">
import { NdArray } from "ndarray";

const a = NdArray.from(
  new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
);
a.free();
</script>

<template>hello</template>
