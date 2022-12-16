import React, { useState, useEffect } from 'react';
import _ from 'lodash';
// import paginate from 'paginate-array';
import axios from 'axios';
import url from './URL';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@material-ui/core';
// import { Pagination } from 'react-paginate';


const Home = () => {
  const [data, setData] = useState();
  const [currentPage, setCurrentPage] = useState(0);
  const ITEMS_PER_PAGE = 10;

  
  useEffect(() => {
    axios.get(url+'/admin/events')
      .then(response => {
        console.log(response.data.events);
        setData(response.data.events);

      })
      .catch(err=> console.error(err));
  },[]);

  return (
    <>
      <TableContainer>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>ID</TableCell>
            <TableCell>Title</TableCell>
            <TableCell>Date</TableCell>
            <TableCell>Venue</TableCell>
            <TableCell>Price</TableCell>
            <TableCell>Description</TableCell>
            <TableCell>Presenter</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data && data.map((row) => (
            <TableRow key={row.id}>
              <TableCell>{row.id}</TableCell>
              <TableCell>{row.title}</TableCell>
              <TableCell>{row.date}</TableCell>
              <TableCell>{row.venue}</TableCell>
              <TableCell>{row.price}</TableCell>
              <TableCell>{row.description}</TableCell>
              <TableCell>{row.presenter}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  </>
  );
}

export default Home