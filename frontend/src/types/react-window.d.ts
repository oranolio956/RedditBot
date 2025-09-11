/**
 * TypeScript definitions for react-window
 * Custom type definitions for virtual scrolling components
 */

declare module 'react-window' {
  import { Component, ReactElement, CSSProperties, Ref } from 'react';

  export interface ListChildComponentProps<T = any> {
    index: number;
    style: CSSProperties;
    data: T;
  }

  export interface FixedSizeListProps {
    children: (props: ListChildComponentProps) => ReactElement;
    height: number;
    itemCount: number;
    itemSize: number;
    itemData?: any;
    className?: string;
    direction?: 'ltr' | 'rtl';
    initialScrollOffset?: number;
    onItemsRendered?: (props: {
      overscanStartIndex: number;
      overscanStopIndex: number;
      visibleStartIndex: number;
      visibleStopIndex: number;
    }) => void;
    onScroll?: (props: {
      scrollDirection: 'forward' | 'backward';
      scrollOffset: number;
      scrollUpdateWasRequested: boolean;
    }) => void;
    overscanCount?: number;
    useIsScrolling?: boolean;
    width?: number | string;
    layout?: 'vertical' | 'horizontal';
  }

  export interface VariableSizeListProps extends Omit<FixedSizeListProps, 'itemSize'> {
    itemSize: (index: number) => number;
    estimatedItemSize?: number;
  }

  export interface ListRef {
    scrollTo(scrollOffset: number): void;
    scrollToItem(index: number, align?: 'auto' | 'smart' | 'center' | 'end' | 'start'): void;
  }

  export class FixedSizeList extends Component<FixedSizeListProps> {
    scrollTo(scrollOffset: number): void;
    scrollToItem(index: number, align?: 'auto' | 'smart' | 'center' | 'end' | 'start'): void;
  }

  export class VariableSizeList extends Component<VariableSizeListProps> {
    scrollTo(scrollOffset: number): void;
    scrollToItem(index: number, align?: 'auto' | 'smart' | 'center' | 'end' | 'start'): void;
    resetAfterIndex(index: number, shouldForceUpdate?: boolean): void;
  }

  export interface FixedSizeGridProps {
    children: (props: {
      columnIndex: number;
      rowIndex: number;
      style: CSSProperties;
      data: any;
    }) => ReactElement;
    columnCount: number;
    columnWidth: number;
    height: number;
    rowCount: number;
    rowHeight: number;
    width: number;
    itemData?: any;
    className?: string;
    direction?: 'ltr' | 'rtl';
    initialScrollLeft?: number;
    initialScrollTop?: number;
    onItemsRendered?: (props: {
      overscanColumnStartIndex: number;
      overscanColumnStopIndex: number;
      overscanRowStartIndex: number;
      overscanRowStopIndex: number;
      visibleColumnStartIndex: number;
      visibleColumnStopIndex: number;
      visibleRowStartIndex: number;
      visibleRowStopIndex: number;
    }) => void;
    onScroll?: (props: {
      horizontalScrollDirection: 'forward' | 'backward';
      scrollLeft: number;
      scrollTop: number;
      scrollUpdateWasRequested: boolean;
      verticalScrollDirection: 'forward' | 'backward';
    }) => void;
    overscanColumnsCount?: number;
    overscanRowsCount?: number;
    useIsScrolling?: boolean;
  }

  export class FixedSizeGrid extends Component<FixedSizeGridProps> {
    scrollTo(props: { scrollLeft: number; scrollTop: number }): void;
    scrollToItem(props: {
      columnIndex?: number;
      rowIndex?: number;
      align?: 'auto' | 'smart' | 'center' | 'end' | 'start';
    }): void;
  }

  export interface VariableSizeGridProps extends Omit<FixedSizeGridProps, 'columnWidth' | 'rowHeight'> {
    columnWidth: (index: number) => number;
    rowHeight: (index: number) => number;
    estimatedColumnWidth?: number;
    estimatedRowHeight?: number;
  }

  export class VariableSizeGrid extends Component<VariableSizeGridProps> {
    scrollTo(props: { scrollLeft: number; scrollTop: number }): void;
    scrollToItem(props: {
      columnIndex?: number;
      rowIndex?: number;
      align?: 'auto' | 'smart' | 'center' | 'end' | 'start';
    }): void;
    resetAfterColumnIndex(index: number, shouldForceUpdate?: boolean): void;
    resetAfterRowIndex(index: number, shouldForceUpdate?: boolean): void;
    resetAfterIndices(props: {
      columnIndex: number;
      rowIndex: number;
      shouldForceUpdate?: boolean;
    }): void;
  }

  export function areEqual<T>(
    prevProps: ListChildComponentProps<T>,
    nextProps: ListChildComponentProps<T>
  ): boolean;

  export function shouldComponentUpdate<T>(
    nextProps: ListChildComponentProps<T>,
    nextState: any
  ): boolean;
}