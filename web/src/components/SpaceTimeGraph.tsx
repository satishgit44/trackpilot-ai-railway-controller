import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { SpaceTimeGraphProps, TrainPath, SpaceTimePoint, ConflictArea, SectionInfo } from '@/types';

interface GraphDimensions {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
}

const SpaceTimeGraph: React.FC<SpaceTimeGraphProps> = ({
  data,
  sections,
  timeRange,
  width,
  height,
  onTrainClick,
  onTimeRangeChange,
  conflicts = []
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState<GraphDimensions>({
    width: width || 800,
    height: height || 600,
    margin: { top: 20, right: 80, bottom: 60, left: 100 }
  });

  // Update dimensions when props change
  useEffect(() => {
    setDimensions(prev => ({
      ...prev,
      width: width || 800,
      height: height || 600
    }));
  }, [width, height]);

  // Calculate inner dimensions
  const innerWidth = dimensions.width - dimensions.margin.left - dimensions.margin.right;
  const innerHeight = dimensions.height - dimensions.margin.top - dimensions.margin.bottom;

  // Create scales
  const xScale = d3.scaleTime()
    .domain([timeRange.start, timeRange.end])
    .range([0, innerWidth]);

  // Calculate cumulative distances for sections
  const sectionPositions = React.useMemo(() => {
    let cumulativeDistance = 0;
    const positions: Record<string, { start: number; end: number; label: string }> = {};
    
    sections.forEach((section, index) => {
      positions[section.section_id] = {
        start: cumulativeDistance,
        end: cumulativeDistance + section.length_km,
        label: section.section_id
      };
      cumulativeDistance += section.length_km;
    });
    
    return { positions, totalDistance: cumulativeDistance };
  }, [sections]);

  const yScale = d3.scaleLinear()
    .domain([0, sectionPositions.totalDistance])
    .range([innerHeight, 0]);

  // Color scale for trains
  const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

  // Line generator for train paths
  const lineGenerator = d3.line<SpaceTimePoint>()
    .x(d => xScale(d.time))
    .y(d => {
      const section = sectionPositions.positions[d.section_id];
      if (!section) return 0;
      return yScale(section.start + (d.position * (section.end - section.start)));
    })
    .curve(d3.curveLinear);

  // Zoom behavior
  const zoomBehavior = React.useMemo(() => {
    return d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        const { transform } = event;
        
        // Update scales with zoom transform
        const newXScale = transform.rescaleX(xScale);
        const newYScale = transform.rescaleY(yScale);
        
        // Update the graph with new scales
        const svg = d3.select(svgRef.current);
        const g = svg.select('g.main-group');
        
        // Update axes
        g.select<SVGGElement>('.x-axis')
          .call(d3.axisBottom(newXScale).tickFormat(d3.timeFormat('%H:%M')) as any);
        
        g.select<SVGGElement>('.y-axis')
          .call(d3.axisLeft(newYScale).tickFormat(d => `${d.toFixed(1)} km`) as any);
        
        // Update train paths
        g.selectAll<SVGPathElement, TrainPath>('.train-path')
          .attr('d', d => {
            const scaledLineGenerator = d3.line<SpaceTimePoint>()
              .x(point => newXScale(point.time))
              .y(point => {
                const section = sectionPositions.positions[point.section_id];
                if (!section) return 0;
                return newYScale(section.start + (point.position * (section.end - section.start)));
              })
              .curve(d3.curveLinear);
            return scaledLineGenerator(d.points) || '';
          });
        
        // Update conflict areas
        g.selectAll<SVGRectElement, ConflictArea>('.conflict-area')
          .attr('x', d => newXScale(d.time_start))
          .attr('width', d => Math.max(0, newXScale(d.time_end) - newXScale(d.time_start)))
          .attr('y', d => {
            const section = sectionPositions.positions[d.section_id];
            return section ? newYScale(section.end) : 0;
          })
          .attr('height', d => {
            const section = sectionPositions.positions[d.section_id];
            return section ? newYScale(section.start) - newYScale(section.end) : 0;
          });
        
        // Notify parent component of time range changes if panning horizontally
        if (onTimeRangeChange && Math.abs(transform.x) > 10) {
          const newDomain = newXScale.domain();
          onTimeRangeChange({ start: newDomain[0], end: newDomain[1] });
        }
      });
  }, [xScale, yScale, sectionPositions, onTimeRangeChange]);

  // Draw the graph
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous content

    // Set up SVG dimensions
    svg
      .attr('width', dimensions.width)
      .attr('height', dimensions.height);

    // Create main group with margins
    const g = svg.append('g')
      .attr('class', 'main-group')
      .attr('transform', `translate(${dimensions.margin.left},${dimensions.margin.top})`);

    // Add clipping path for the main drawing area
    svg.append('defs')
      .append('clipPath')
      .attr('id', 'clip')
      .append('rect')
      .attr('width', innerWidth)
      .attr('height', innerHeight);

    // Create background
    g.append('rect')
      .attr('class', 'background')
      .attr('width', innerWidth)
      .attr('height', innerHeight)
      .attr('fill', '#f8f9fa')
      .attr('stroke', '#e9ecef')
      .attr('stroke-width', 1);

    // Draw section dividers
    sections.forEach((section, index) => {
      const position = sectionPositions.positions[section.section_id];
      if (!position) return;

      // Section divider line
      g.append('line')
        .attr('class', 'section-divider')
        .attr('x1', 0)
        .attr('x2', innerWidth)
        .attr('y1', yScale(position.start))
        .attr('y2', yScale(position.start))
        .attr('stroke', '#dee2e6')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');

      // Section label
      g.append('text')
        .attr('class', 'section-label')
        .attr('x', -5)
        .attr('y', yScale((position.start + position.end) / 2))
        .attr('dy', '0.35em')
        .attr('text-anchor', 'end')
        .attr('font-size', '12px')
        .attr('fill', '#6c757d')
        .text(section.section_id);

      // Track capacity indicator
      if (section.is_single_track) {
        g.append('rect')
          .attr('class', 'single-track-indicator')
          .attr('x', -10)
          .attr('y', yScale(position.end))
          .attr('width', 5)
          .attr('height', yScale(position.start) - yScale(position.end))
          .attr('fill', '#ffc107')
          .attr('opacity', 0.6);
      }
    });

    // Draw conflict areas
    g.selectAll('.conflict-area')
      .data(conflicts)
      .enter()
      .append('rect')
      .attr('class', 'conflict-area')
      .attr('x', d => xScale(d.time_start))
      .attr('width', d => Math.max(0, xScale(d.time_end) - xScale(d.time_start)))
      .attr('y', d => {
        const section = sectionPositions.positions[d.section_id];
        return section ? yScale(section.end) : 0;
      })
      .attr('height', d => {
        const section = sectionPositions.positions[d.section_id];
        return section ? yScale(section.start) - yScale(section.end) : 0;
      })
      .attr('fill', d => {
        switch (d.severity) {
          case 'high': return '#dc3545';
          case 'medium': return '#fd7e14';
          case 'low': return '#ffc107';
          default: return '#6c757d';
        }
      })
      .attr('opacity', 0.3)
      .attr('clip-path', 'url(#clip)');

    // Draw train paths
    const trainPaths = g.selectAll('.train-path')
      .data(data)
      .enter()
      .append('path')
      .attr('class', 'train-path')
      .attr('d', d => lineGenerator(d.points) || '')
      .attr('stroke', d => colorScale(d.train_id))
      .attr('stroke-width', d => d.priority > 7 ? 3 : 2)
      .attr('fill', 'none')
      .attr('opacity', 0.8)
      .attr('clip-path', 'url(#clip)')
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        onTrainClick?.(d.train_id);
      })
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke-width', d.priority > 7 ? 5 : 4)
          .attr('opacity', 1);
        
        // Show tooltip
        const tooltip = d3.select('body')
          .append('div')
          .attr('class', 'train-tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0,0,0,0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            <strong>${d.train_id}</strong><br/>
            Priority: ${d.priority}<br/>
            Type: ${d.is_freight ? 'Freight' : 'Passenger'}<br/>
            Points: ${d.points.length}
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .attr('stroke-width', d.priority > 7 ? 3 : 2)
          .attr('opacity', 0.8);
        
        // Remove tooltip
        d3.selectAll('.train-tooltip').remove();
      });

    // Draw train markers (current positions)
    data.forEach(trainPath => {
      const currentPoint = trainPath.points[trainPath.points.length - 1];
      if (currentPoint && currentPoint.status === 'actual') {
        const section = sectionPositions.positions[currentPoint.section_id];
        if (section) {
          const x = xScale(currentPoint.time);
          const y = yScale(section.start + (currentPoint.position * (section.end - section.start)));
          
          g.append('circle')
            .attr('class', 'train-marker')
            .attr('cx', x)
            .attr('cy', y)
            .attr('r', trainPath.priority > 7 ? 6 : 4)
            .attr('fill', colorScale(trainPath.train_id))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .attr('clip-path', 'url(#clip)');
        }
      }
    });

    // Create axes
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.timeFormat('%H:%M'))
      .ticks(10);

    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d => `${d.toFixed(1)} km`)
      .ticks(8);

    // Add X axis
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis);

    // Add Y axis
    g.append('g')
      .attr('class', 'y-axis')
      .call(yAxis);

    // Add axis labels
    g.append('text')
      .attr('class', 'x-axis-label')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 45)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', '#495057')
      .text('Time');

    g.append('text')
      .attr('class', 'y-axis-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -60)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', '#495057')
      .text('Distance (km)');

    // Apply zoom behavior
    svg.call(zoomBehavior);

    // Add legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${dimensions.width - 70}, 30)`);

    const legendItems = [
      { label: 'High Priority', color: '#dc3545', width: 3 },
      { label: 'Normal', color: '#007bff', width: 2 },
      { label: 'Freight', color: '#28a745', width: 2 }
    ];

    legendItems.forEach((item, index) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${index * 20})`);

      legendItem.append('line')
        .attr('x1', 0)
        .attr('x2', 20)
        .attr('y1', 0)
        .attr('y2', 0)
        .attr('stroke', item.color)
        .attr('stroke-width', item.width);

      legendItem.append('text')
        .attr('x', 25)
        .attr('y', 0)
        .attr('dy', '0.35em')
        .attr('font-size', '12px')
        .attr('fill', '#495057')
        .text(item.label);
    });

  }, [data, sections, timeRange, dimensions, xScale, yScale, sectionPositions, colorScale, lineGenerator, conflicts, onTrainClick, zoomBehavior]);

  return (
    <div className="space-time-graph">
      <svg
        ref={svgRef}
        style={{
          width: '100%',
          height: '100%',
          border: '1px solid #e9ecef',
          borderRadius: '4px'
        }}
      />
    </div>
  );
};

export default React.memo(SpaceTimeGraph);